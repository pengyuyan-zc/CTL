"""
Federated Learning Framework for CTN-LT
实现联邦对比学习和个性化双编码器
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy
from collections import OrderedDict
import numpy as np

from ..models.ctn_lt import CTN_LT, CTN_LT_Loss


class FederatedClient:
    """
    联邦学习客户端
    每个客户端维护本地模型和数据
    """
    def __init__(
        self,
        client_id: int,
        model: CTN_LT,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = 'cuda',
        learning_rate: float = 5e-5,
        alpha: float = 0.8,
        m: int = 50,
        personalized: bool = True
    ):
        self.client_id = client_id
        self.device = device
        self.personalized = personalized
        
        # 模型
        self.model = model.to(device)
        
        # 数据加载器
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # 损失函数
        self.criterion = CTN_LT_Loss(alpha=alpha, m=m)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # 全局标签嵌入池（用于负采样）
        self.global_label_embeddings = None
        self.global_label_ids = None
        
    def set_global_label_pool(
        self,
        label_embeddings: torch.Tensor,
        label_ids: List[int]
    ):
        """
        设置全局标签嵌入池
        Args:
            label_embeddings: [num_global_labels, hidden_size]
            label_ids: 标签ID列表
        """
        self.global_label_embeddings = label_embeddings.to(self.device)
        self.global_label_ids = label_ids
        
    def local_train(
        self,
        num_epochs: int = 1,
        use_global_negatives: bool = True,
        global_negative_ratio: float = 0.3
    ) -> Dict[str, float]:
        """
        本地训练
        Args:
            num_epochs: 训练轮数
            use_global_negatives: 是否使用全局负样本
            global_negative_ratio: 全局负样本比例
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        
        total_loss = 0.0
        total_ce_loss = 0.0
        total_mbce_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch in self.train_loader:
                # 获取批次数据
                doc_input_ids = batch['doc_input_ids'].to(self.device)
                doc_attention_mask = batch['doc_attention_mask'].to(self.device)
                label_input_ids = batch['label_input_ids'].to(self.device)
                label_attention_mask = batch['label_attention_mask'].to(self.device)
                targets = batch['targets'].to(self.device)  # [batch_size, num_labels_in_batch]
                
                # 前向传播
                doc_emb, label_emb, similarity = self.model(
                    doc_input_ids, doc_attention_mask,
                    label_input_ids, label_attention_mask
                )
                
                # 如果使用全局负样本，扩展相似度矩阵
                if use_global_negatives and self.global_label_embeddings is not None:
                    # 采样全局负样本
                    num_global_negatives = int(label_emb.size(0) * global_negative_ratio)
                    if num_global_negatives > 0:
                        # 随机采样
                        indices = torch.randperm(self.global_label_embeddings.size(0))[:num_global_negatives]
                        global_neg_emb = self.global_label_embeddings[indices]
                        
                        # 计算与全局负样本的相似度
                        global_similarity = torch.matmul(doc_emb, global_neg_emb.t())
                        
                        # 扩展相似度矩阵和目标矩阵
                        similarity = torch.cat([similarity, global_similarity], dim=1)
                        global_targets = torch.zeros(
                            targets.size(0), num_global_negatives,
                            device=self.device
                        )
                        targets = torch.cat([targets, global_targets], dim=1)
                
                # 计算损失
                loss, loss_dict = self.criterion(similarity, targets)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 记录损失
                total_loss += loss_dict['total_loss']
                total_ce_loss += loss_dict['ce_loss']
                total_mbce_loss += loss_dict['mbce_loss']
                num_batches += 1
        
        metrics = {
            'loss': total_loss / num_batches,
            'ce_loss': total_ce_loss / num_batches,
            'mbce_loss': total_mbce_loss / num_batches
        }
        
        return metrics
    
    def get_model_parameters(self) -> OrderedDict:
        """获取模型参数"""
        return copy.deepcopy(self.model.state_dict())
    
    def set_model_parameters(self, parameters: OrderedDict):
        """设置模型参数"""
        self.model.load_state_dict(parameters)
    
    def get_label_encoder_parameters(self) -> OrderedDict:
        """获取标签编码器参数（用于全局共享）"""
        return copy.deepcopy(self.model.label_encoder.state_dict())
    
    def set_label_encoder_parameters(self, parameters: OrderedDict):
        """设置标签编码器参数"""
        self.model.label_encoder.load_state_dict(parameters)
    
    def get_document_encoder_parameters(self) -> OrderedDict:
        """获取文档编码器参数（用于个性化）"""
        return copy.deepcopy(self.model.document_encoder.state_dict())
    
    def set_document_encoder_parameters(self, parameters: OrderedDict):
        """设置文档编码器参数"""
        self.model.document_encoder.load_state_dict(parameters)


class FederatedServer:
    """
    联邦学习服务器
    负责聚合客户端模型和维护全局标签池
    """
    def __init__(
        self,
        global_model: CTN_LT,
        device: str = 'cuda',
        aggregation_method: str = 'fedavg'
    ):
        self.global_model = global_model.to(device)
        self.device = device
        self.aggregation_method = aggregation_method
        
        # 全局标签嵌入池
        self.global_label_embeddings = None
        self.global_label_ids = None
        
    def aggregate_models(
        self,
        client_parameters: List[OrderedDict],
        client_weights: Optional[List[float]] = None,
        aggregate_label_encoder: bool = True,
        aggregate_document_encoder: bool = False
    ) -> OrderedDict:
        """
        聚合客户端模型
        Args:
            client_parameters: 客户端参数列表
            client_weights: 客户端权重（默认均等）
            aggregate_label_encoder: 是否聚合标签编码器
            aggregate_document_encoder: 是否聚合文档编码器
        Returns:
            aggregated_parameters: 聚合后的参数
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_parameters)] * len(client_parameters)
        
        # 归一化权重
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # 初始化聚合参数
        aggregated_parameters = OrderedDict()
        
        # 获取第一个客户端的参数作为模板
        for key in client_parameters[0].keys():
            # 判断是否需要聚合该参数
            should_aggregate = False
            
            if aggregate_label_encoder and 'label_encoder' in key:
                should_aggregate = True
            if aggregate_document_encoder and 'document_encoder' in key:
                should_aggregate = True
            
            if should_aggregate:
                # 加权平均
                aggregated_parameters[key] = sum(
                    client_weights[i] * client_parameters[i][key]
                    for i in range(len(client_parameters))
                )
            else:
                # 不聚合，保持全局模型的参数
                aggregated_parameters[key] = self.global_model.state_dict()[key]
        
        return aggregated_parameters
    
    def update_global_label_pool(
        self,
        label_texts: List[str],
        tokenizer,
        max_length: int = 16
    ):
        """
        更新全局标签嵌入池
        Args:
            label_texts: 标签文本列表
            tokenizer: 分词器
            max_length: 最大长度
        """
        self.global_model.eval()
        
        # 编码标签文本
        encoded = tokenizer(
            label_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        # 生成标签嵌入
        with torch.no_grad():
            label_embeddings = self.global_model.encode_labels(input_ids, attention_mask)
        
        self.global_label_embeddings = label_embeddings.cpu()
        self.global_label_ids = list(range(len(label_texts)))
        
    def get_global_label_pool(self) -> Tuple[torch.Tensor, List[int]]:
        """获取全局标签池"""
        return self.global_label_embeddings, self.global_label_ids
    
    def get_global_model_parameters(self) -> OrderedDict:
        """获取全局模型参数"""
        return copy.deepcopy(self.global_model.state_dict())
    
    def set_global_model_parameters(self, parameters: OrderedDict):
        """设置全局模型参数"""
        self.global_model.load_state_dict(parameters)


class FederatedTrainer:
    """
    联邦训练器
    协调服务器和客户端进行联邦学习
    """
    def __init__(
        self,
        server: FederatedServer,
        clients: List[FederatedClient],
        num_rounds: int = 100,
        clients_per_round: int = 10,
        local_epochs: int = 1,
        use_global_negatives: bool = True,
        global_negative_ratio: float = 0.3,
        personalized: bool = True
    ):
        self.server = server
        self.clients = clients
        self.num_rounds = num_rounds
        self.clients_per_round = min(clients_per_round, len(clients))
        self.local_epochs = local_epochs
        self.use_global_negatives = use_global_negatives
        self.global_negative_ratio = global_negative_ratio
        self.personalized = personalized
        
    def train(self) -> Dict[str, List[float]]:
        """
        执行联邦训练
        Returns:
            history: 训练历史
        """
        history = {
            'round': [],
            'avg_loss': [],
            'avg_ce_loss': [],
            'avg_mbce_loss': []
        }
        
        for round_idx in range(self.num_rounds):
            print(f"\n=== Round {round_idx + 1}/{self.num_rounds} ===")
            
            # 选择客户端
            selected_clients = np.random.choice(
                self.clients,
                size=self.clients_per_round,
                replace=False
            )
            
            # 分发全局模型
            global_params = self.server.get_global_model_parameters()
            for client in selected_clients:
                if self.personalized:
                    # 个性化：只更新标签编码器
                    client.set_label_encoder_parameters(
                        {k: v for k, v in global_params.items() if 'label_encoder' in k}
                    )
                else:
                    # 非个性化：更新整个模型
                    client.set_model_parameters(global_params)
            
            # 分发全局标签池
            if self.use_global_negatives:
                label_emb, label_ids = self.server.get_global_label_pool()
                for client in selected_clients:
                    client.set_global_label_pool(label_emb, label_ids)
            
            # 本地训练
            client_metrics = []
            client_parameters = []
            
            for client in selected_clients:
                print(f"Training client {client.client_id}...")
                metrics = client.local_train(
                    num_epochs=self.local_epochs,
                    use_global_negatives=self.use_global_negatives,
                    global_negative_ratio=self.global_negative_ratio
                )
                client_metrics.append(metrics)
                
                # 收集参数
                if self.personalized:
                    # 只收集标签编码器参数
                    params = client.get_label_encoder_parameters()
                else:
                    # 收集全部参数
                    params = client.get_model_parameters()
                
                client_parameters.append(params)
                
                print(f"  Loss: {metrics['loss']:.4f}, "
                      f"CE: {metrics['ce_loss']:.4f}, "
                      f"mBCE: {metrics['mbce_loss']:.4f}")
            
            # 聚合模型
            aggregated_params = self.server.aggregate_models(
                client_parameters,
                aggregate_label_encoder=True,
                aggregate_document_encoder=not self.personalized
            )
            self.server.set_global_model_parameters(aggregated_params)
            
            # 记录平均指标
            avg_loss = np.mean([m['loss'] for m in client_metrics])
            avg_ce_loss = np.mean([m['ce_loss'] for m in client_metrics])
            avg_mbce_loss = np.mean([m['mbce_loss'] for m in client_metrics])
            
            history['round'].append(round_idx + 1)
            history['avg_loss'].append(avg_loss)
            history['avg_ce_loss'].append(avg_ce_loss)
            history['avg_mbce_loss'].append(avg_mbce_loss)
            
            print(f"Round {round_idx + 1} - Avg Loss: {avg_loss:.4f}, "
                  f"CE: {avg_ce_loss:.4f}, mBCE: {avg_mbce_loss:.4f}")
        
        return history


if __name__ == "__main__":
    print("Federated Learning Framework Test")
    print("This module should be imported and used with proper data loaders.")
