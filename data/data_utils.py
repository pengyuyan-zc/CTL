"""
Data processing module for Federated CTN-LT
处理多标签文本分类数据集并模拟联邦场景
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from collections import defaultdict


class MultiLabelTextDataset(Dataset):
    """
    多标签文本分类数据集
    """
    def __init__(
        self,
        documents: List[str],
        labels: List[List[int]],  # 每个文档的标签ID列表
        label_texts: List[str],   # 所有标签的文本描述
        tokenizer,
        doc_max_length: int = 256,
        label_max_length: int = 16
    ):
        self.documents = documents
        self.labels = labels
        self.label_texts = label_texts
        self.tokenizer = tokenizer
        self.doc_max_length = doc_max_length
        self.label_max_length = label_max_length
        
        # 构建标签ID到索引的映射
        self.num_labels = len(label_texts)
        
    def __len__(self):
        return len(self.documents)
    
    def __getitem__(self, idx):
        # 获取文档
        document = self.documents[idx]
        doc_label_ids = self.labels[idx]
        
        # 编码文档
        doc_encoded = self.tokenizer(
            document,
            padding='max_length',
            truncation=True,
            max_length=self.doc_max_length,
            return_tensors='pt'
        )
        
        # 获取该文档的标签文本
        doc_label_texts = [self.label_texts[lid] for lid in doc_label_ids]
        
        # 编码标签
        if len(doc_label_texts) > 0:
            label_encoded = self.tokenizer(
                doc_label_texts,
                padding='max_length',
                truncation=True,
                max_length=self.label_max_length,
                return_tensors='pt'
            )
        else:
            # 如果没有标签，创建空张量
            label_encoded = {
                'input_ids': torch.zeros((1, self.label_max_length), dtype=torch.long),
                'attention_mask': torch.zeros((1, self.label_max_length), dtype=torch.long)
            }
        
        # 创建目标向量（二值向量）
        targets = torch.zeros(len(doc_label_ids), dtype=torch.float)
        targets[:len(doc_label_ids)] = 1.0
        
        return {
            'doc_input_ids': doc_encoded['input_ids'].squeeze(0),
            'doc_attention_mask': doc_encoded['attention_mask'].squeeze(0),
            'label_input_ids': label_encoded['input_ids'],
            'label_attention_mask': label_encoded['attention_mask'],
            'targets': targets,
            'label_ids': torch.tensor(doc_label_ids, dtype=torch.long)
        }


def collate_fn(batch):
    """
    自定义collate函数，处理变长的标签
    """
    # 收集所有批次中的唯一标签
    all_label_ids = []
    for item in batch:
        all_label_ids.extend(item['label_ids'].tolist())
    
    unique_label_ids = sorted(list(set(all_label_ids)))
    label_id_to_idx = {lid: idx for idx, lid in enumerate(unique_label_ids)}
    
    # 准备批次数据
    batch_size = len(batch)
    num_unique_labels = len(unique_label_ids)
    
    doc_input_ids = torch.stack([item['doc_input_ids'] for item in batch])
    doc_attention_mask = torch.stack([item['doc_attention_mask'] for item in batch])
    
    # 收集唯一标签的编码
    label_input_ids_list = []
    label_attention_mask_list = []
    
    for lid in unique_label_ids:
        # 找到第一个包含该标签的样本
        for item in batch:
            if lid in item['label_ids'].tolist():
                idx_in_item = (item['label_ids'] == lid).nonzero(as_tuple=True)[0][0]
                label_input_ids_list.append(item['label_input_ids'][idx_in_item])
                label_attention_mask_list.append(item['label_attention_mask'][idx_in_item])
                break
    
    label_input_ids = torch.stack(label_input_ids_list)
    label_attention_mask = torch.stack(label_attention_mask_list)
    
    # 创建目标矩阵
    targets = torch.zeros(batch_size, num_unique_labels)
    for i, item in enumerate(batch):
        for lid in item['label_ids'].tolist():
            targets[i, label_id_to_idx[lid]] = 1.0
    
    return {
        'doc_input_ids': doc_input_ids,
        'doc_attention_mask': doc_attention_mask,
        'label_input_ids': label_input_ids,
        'label_attention_mask': label_attention_mask,
        'targets': targets,
        'unique_label_ids': unique_label_ids
    }


class FederatedDataPartitioner:
    """
    联邦数据分区器
    将数据集划分为多个客户端，模拟Non-IID场景
    """
    def __init__(
        self,
        documents: List[str],
        labels: List[List[int]],
        label_texts: List[str],
        num_clients: int = 10,
        partition_method: str = 'label_distribution',
        alpha: float = 0.5  # Dirichlet分布参数
    ):
        self.documents = documents
        self.labels = labels
        self.label_texts = label_texts
        self.num_clients = num_clients
        self.partition_method = partition_method
        self.alpha = alpha
        
        # 执行分区
        self.client_indices = self._partition_data()
        
    def _partition_data(self) -> List[List[int]]:
        """
        根据分区方法划分数据
        """
        if self.partition_method == 'iid':
            return self._partition_iid()
        elif self.partition_method == 'label_distribution':
            return self._partition_by_label_distribution()
        elif self.partition_method == 'dirichlet':
            return self._partition_dirichlet()
        else:
            raise ValueError(f"Unknown partition method: {self.partition_method}")
    
    def _partition_iid(self) -> List[List[int]]:
        """IID划分：随机均匀分配"""
        num_samples = len(self.documents)
        indices = np.random.permutation(num_samples)
        
        client_indices = []
        samples_per_client = num_samples // self.num_clients
        
        for i in range(self.num_clients):
            start = i * samples_per_client
            end = start + samples_per_client if i < self.num_clients - 1 else num_samples
            client_indices.append(indices[start:end].tolist())
        
        return client_indices
    
    def _partition_by_label_distribution(self) -> List[List[int]]:
        """
        按标签分布划分：每个客户端专注于某些标签
        模拟不同机构拥有不同类型数据的场景
        """
        # 统计每个标签的样本
        label_to_samples = defaultdict(list)
        for idx, label_list in enumerate(self.labels):
            for label_id in label_list:
                label_to_samples[label_id].append(idx)
        
        # 为每个客户端分配主要标签
        all_label_ids = list(label_to_samples.keys())
        labels_per_client = len(all_label_ids) // self.num_clients
        
        client_indices = [[] for _ in range(self.num_clients)]
        
        for client_id in range(self.num_clients):
            # 选择该客户端的主要标签
            start_label = client_id * labels_per_client
            end_label = start_label + labels_per_client if client_id < self.num_clients - 1 else len(all_label_ids)
            client_labels = all_label_ids[start_label:end_label]
            
            # 收集包含这些标签的样本
            client_samples = set()
            for label_id in client_labels:
                client_samples.update(label_to_samples[label_id])
            
            client_indices[client_id] = list(client_samples)
        
        return client_indices
    
    def _partition_dirichlet(self) -> List[List[int]]:
        """
        Dirichlet分布划分：更真实的Non-IID场景
        """
        num_samples = len(self.documents)
        
        # 为每个标签使用Dirichlet分布分配样本到客户端
        label_to_samples = defaultdict(list)
        for idx, label_list in enumerate(self.labels):
            for label_id in label_list:
                label_to_samples[label_id].append(idx)
        
        client_indices = [[] for _ in range(self.num_clients)]
        
        for label_id, sample_indices in label_to_samples.items():
            # 使用Dirichlet分布生成分配比例
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # 根据比例分配样本
            num_label_samples = len(sample_indices)
            sample_indices = np.array(sample_indices)
            np.random.shuffle(sample_indices)
            
            start_idx = 0
            for client_id in range(self.num_clients):
                num_client_samples = int(proportions[client_id] * num_label_samples)
                end_idx = start_idx + num_client_samples
                
                if client_id == self.num_clients - 1:
                    end_idx = num_label_samples
                
                client_indices[client_id].extend(sample_indices[start_idx:end_idx].tolist())
                start_idx = end_idx
        
        # 去重并打乱
        for i in range(self.num_clients):
            client_indices[i] = list(set(client_indices[i]))
            np.random.shuffle(client_indices[i])
        
        return client_indices
    
    def get_client_data(
        self,
        client_id: int,
        tokenizer,
        doc_max_length: int = 256,
        label_max_length: int = 16
    ) -> Tuple[List[str], List[List[int]]]:
        """
        获取指定客户端的数据
        """
        indices = self.client_indices[client_id]
        
        client_documents = [self.documents[i] for i in indices]
        client_labels = [self.labels[i] for i in indices]
        
        return client_documents, client_labels
    
    def create_client_dataloaders(
        self,
        tokenizer,
        batch_size: int = 32,
        doc_max_length: int = 256,
        label_max_length: int = 16,
        train_split: float = 0.9
    ) -> List[Tuple[DataLoader, DataLoader]]:
        """
        为所有客户端创建数据加载器
        Returns:
            List of (train_loader, val_loader) tuples
        """
        client_loaders = []
        
        for client_id in range(self.num_clients):
            # 获取客户端数据
            client_docs, client_labels = self.get_client_data(
                client_id, tokenizer, doc_max_length, label_max_length
            )
            
            # 划分训练集和验证集
            num_samples = len(client_docs)
            num_train = int(num_samples * train_split)
            
            train_docs = client_docs[:num_train]
            train_labels = client_labels[:num_train]
            val_docs = client_docs[num_train:]
            val_labels = client_labels[num_train:]
            
            # 创建数据集
            train_dataset = MultiLabelTextDataset(
                train_docs, train_labels, self.label_texts,
                tokenizer, doc_max_length, label_max_length
            )
            
            val_dataset = MultiLabelTextDataset(
                val_docs, val_labels, self.label_texts,
                tokenizer, doc_max_length, label_max_length
            ) if len(val_docs) > 0 else None
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn
            ) if val_dataset is not None else None
            
            client_loaders.append((train_loader, val_loader))
        
        return client_loaders
    
    def get_statistics(self) -> Dict:
        """获取数据分区统计信息"""
        stats = {
            'num_clients': self.num_clients,
            'partition_method': self.partition_method,
            'client_sizes': [len(indices) for indices in self.client_indices],
            'total_samples': len(self.documents)
        }
        
        # 计算每个客户端的标签分布
        client_label_counts = []
        for indices in self.client_indices:
            label_count = defaultdict(int)
            for idx in indices:
                for label_id in self.labels[idx]:
                    label_count[label_id] += 1
            client_label_counts.append(dict(label_count))
        
        stats['client_label_distributions'] = client_label_counts
        
        return stats


def load_dataset_from_json(
    json_path: str
) -> Tuple[List[str], List[List[int]], List[str]]:
    """
    从JSON文件加载数据集
    
    JSON格式:
    {
        "documents": ["text1", "text2", ...],
        "labels": [[0, 1, 2], [1, 3], ...],
        "label_texts": ["label0", "label1", ...]
    }
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data['documents'], data['labels'], data['label_texts']


if __name__ == "__main__":
    # 测试代码
    print("Testing data processing module...")
    
    # 创建虚拟数据
    documents = [f"This is document {i}" for i in range(100)]
    labels = [[i % 10, (i+1) % 10] for i in range(100)]
    label_texts = [f"label_{i}" for i in range(10)]
    
    # 创建分区器
    partitioner = FederatedDataPartitioner(
        documents, labels, label_texts,
        num_clients=5,
        partition_method='label_distribution'
    )
    
    # 获取统计信息
    stats = partitioner.get_statistics()
    print(f"Number of clients: {stats['num_clients']}")
    print(f"Client sizes: {stats['client_sizes']}")
    
    # 创建tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # 创建数据加载器
    client_loaders = partitioner.create_client_dataloaders(
        tokenizer, batch_size=4
    )
    
    print(f"Created {len(client_loaders)} client dataloaders")
    
    # 测试第一个客户端的数据加载器
    train_loader, val_loader = client_loaders[0]
    print(f"Client 0 - Train batches: {len(train_loader)}")
    
    # 测试一个批次
    batch = next(iter(train_loader))
    print(f"Batch keys: {batch.keys()}")
    print(f"Doc input shape: {batch['doc_input_ids'].shape}")
    print(f"Label input shape: {batch['label_input_ids'].shape}")
    print(f"Targets shape: {batch['targets'].shape}")
    
    print("\nData processing test passed!")
