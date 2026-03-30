"""
CTN-LT Model Implementation for Federated Learning
基于论文: Contrastive Transformer Network for Long Tail Classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List, Tuple, Optional


class GatedEmbeddingHead(nn.Module):
    """
    门控嵌入头，用于自适应调整预训练模型的输出
    类似于Adapter层，允许模型跳过更新
    """
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, hidden_size]
        Returns:
            output: [batch_size, hidden_size]
        """
        # 线性变换
        transformed = self.linear(x)
        # 门控机制
        gate_values = torch.sigmoid(self.gate(x))
        # 应用dropout
        gated = self.dropout(gate_values * transformed)
        # 残差连接
        output = x + gated
        return output


class DocumentEncoder(nn.Module):
    """
    文档编码器
    在联邦学习中，每个客户端可以有个性化的文档编码器
    """
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # 可选：冻结预训练模型参数
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.embedding_head = GatedEmbeddingHead(hidden_size, dropout)
        self.hidden_size = hidden_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # 获取[CLS]标记的输出
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # 通过门控头
        embeddings = self.embedding_head(cls_output)
        
        # L2归一化（用于内积相似度计算）
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class LabelEncoder(nn.Module):
    """
    标签编码器
    在联邦学习中，标签编码器通常是全局共享的
    """
    def __init__(
        self, 
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        self.embedding_head = GatedEmbeddingHead(hidden_size, dropout)
        self.hidden_size = hidden_size
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [num_labels, seq_len]
            attention_mask: [num_labels, seq_len]
        Returns:
            embeddings: [num_labels, hidden_size]
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        embeddings = self.embedding_head(cls_output)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings


class CTN_LT(nn.Module):
    """
    完整的CTN-LT模型
    包含文档编码器和标签编码器
    """
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        hidden_size: int = 768,
        dropout: float = 0.1,
        freeze_base: bool = False
    ):
        super().__init__()
        
        self.document_encoder = DocumentEncoder(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_base=freeze_base
        )
        
        self.label_encoder = LabelEncoder(
            model_name=model_name,
            hidden_size=hidden_size,
            dropout=dropout,
            freeze_base=freeze_base
        )
        
        self.hidden_size = hidden_size
        
    def forward(
        self,
        doc_input_ids: torch.Tensor,
        doc_attention_mask: torch.Tensor,
        label_input_ids: torch.Tensor,
        label_attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            doc_input_ids: [batch_size, doc_seq_len]
            doc_attention_mask: [batch_size, doc_seq_len]
            label_input_ids: [num_labels_in_batch, label_seq_len]
            label_attention_mask: [num_labels_in_batch, label_seq_len]
            
        Returns:
            doc_embeddings: [batch_size, hidden_size]
            label_embeddings: [num_labels_in_batch, hidden_size]
            similarity_matrix: [batch_size, num_labels_in_batch]
        """
        # 编码文档
        doc_embeddings = self.document_encoder(doc_input_ids, doc_attention_mask)
        
        # 编码标签
        label_embeddings = self.label_encoder(label_input_ids, label_attention_mask)
        
        # 计算相似度矩阵（内积）
        similarity_matrix = torch.matmul(doc_embeddings, label_embeddings.t())
        
        return doc_embeddings, label_embeddings, similarity_matrix
    
    def encode_documents(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """仅编码文档（推理时使用）"""
        return self.document_encoder(input_ids, attention_mask)
    
    def encode_labels(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """仅编码标签（推理时使用）"""
        return self.label_encoder(input_ids, attention_mask)


class MaskedBCELoss(nn.Module):
    """
    Masked Binary Cross Entropy Loss
    只保留top-m个最大损失，解决类别不平衡问题
    """
    def __init__(self, m: int = 50):
        super().__init__()
        self.m = m
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels] - 相似度分数
            targets: [batch_size, num_labels] - 二值标签
        Returns:
            loss: scalar
        """
        # 应用sigmoid
        probs = torch.sigmoid(logits)
        
        # 计算每个样本的BCE损失
        bce_loss = -targets * torch.log(probs + 1e-8) - (1 - targets) * torch.log(1 - probs + 1e-8)
        # [batch_size, num_labels]
        
        # 对每个样本，选择top-m个最大损失
        batch_size = logits.size(0)
        masked_losses = []
        
        for i in range(batch_size):
            sample_losses = bce_loss[i]  # [num_labels]
            # 选择top-m
            top_m_losses, _ = torch.topk(sample_losses, min(self.m, sample_losses.size(0)))
            masked_losses.append(top_m_losses.mean())
        
        # 平均所有样本的损失
        loss = torch.stack(masked_losses).mean()
        
        return loss


class AdaptedCELoss(nn.Module):
    """
    Adapted Cross Entropy Loss
    为每个正标签动态构建softmax分布
    """
    def __init__(self):
        super().__init__()
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_labels]
            targets: [batch_size, num_labels]
        Returns:
            loss: scalar
        """
        batch_size, num_labels = logits.shape
        total_loss = 0.0
        num_positive = 0
        
        for i in range(batch_size):
            # 找到正标签
            positive_indices = torch.where(targets[i] == 1)[0]
            
            if len(positive_indices) == 0:
                continue
            
            # 对每个正标签
            for pos_idx in positive_indices:
                # 获取该正标签和所有负标签的logits
                pos_logit = logits[i, pos_idx]
                
                # 负标签索引
                neg_mask = targets[i] == 0
                neg_logits = logits[i][neg_mask]
                
                # 组合正标签和负标签的logits
                combined_logits = torch.cat([pos_logit.unsqueeze(0), neg_logits])
                
                # 应用softmax（正标签在第0位）
                log_probs = F.log_softmax(combined_logits, dim=0)
                
                # CE损失（目标是第0位）
                total_loss += -log_probs[0]
                num_positive += 1
        
        if num_positive == 0:
            return torch.tensor(0.0, device=logits.device)
        
        return total_loss / num_positive


class CTN_LT_Loss(nn.Module):
    """
    CTN-LT组合损失函数
    J_CTN-LT = α * J_CE + (1-α) * J_mBCE
    """
    def __init__(self, alpha: float = 0.8, m: int = 50):
        super().__init__()
        self.alpha = alpha
        self.mbce_loss = MaskedBCELoss(m=m)
        self.ce_loss = AdaptedCELoss()
        
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            logits: [batch_size, num_labels]
            targets: [batch_size, num_labels]
        Returns:
            loss: scalar
            loss_dict: 各部分损失的字典
        """
        ce_loss = self.ce_loss(logits, targets)
        mbce_loss = self.mbce_loss(logits, targets)
        
        total_loss = self.alpha * ce_loss + (1 - self.alpha) * mbce_loss
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'ce_loss': ce_loss.item(),
            'mbce_loss': mbce_loss.item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # 测试代码
    print("Testing CTN-LT Model...")
    
    # 创建模型
    model = CTN_LT(model_name="distilbert-base-uncased")
    
    # 创建虚拟数据
    batch_size = 4
    num_labels = 10
    doc_seq_len = 128
    label_seq_len = 16
    
    doc_input_ids = torch.randint(0, 30522, (batch_size, doc_seq_len))
    doc_attention_mask = torch.ones(batch_size, doc_seq_len)
    label_input_ids = torch.randint(0, 30522, (num_labels, label_seq_len))
    label_attention_mask = torch.ones(num_labels, label_seq_len)
    
    # 前向传播
    doc_emb, label_emb, similarity = model(
        doc_input_ids, doc_attention_mask,
        label_input_ids, label_attention_mask
    )
    
    print(f"Document embeddings shape: {doc_emb.shape}")
    print(f"Label embeddings shape: {label_emb.shape}")
    print(f"Similarity matrix shape: {similarity.shape}")
    
    # 测试损失函数
    targets = torch.zeros(batch_size, num_labels)
    targets[0, [0, 1, 2]] = 1  # 第一个样本有3个正标签
    targets[1, [3, 4]] = 1     # 第二个样本有2个正标签
    
    criterion = CTN_LT_Loss(alpha=0.8, m=5)
    loss, loss_dict = criterion(similarity, targets)
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
    
    print("\nModel test passed!")
