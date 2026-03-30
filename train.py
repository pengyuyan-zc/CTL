"""
Main training script for Federated CTN-LT
联邦CTN-LT主训练脚本
"""

import torch
import argparse
import json
import os
from transformers import AutoTokenizer
import numpy as np
from typing import Optional

from federated_ctn_lt.models import CTN_LT
from federated_ctn_lt.federated import FederatedClient, FederatedServer, FederatedTrainer
from federated_ctn_lt.data import FederatedDataPartitioner, load_dataset_from_json
from federated_ctn_lt.evaluation import MetricsCalculator, LabelFrequencyAnalyzer


def parse_args():
    parser = argparse.ArgumentParser(description='Federated CTN-LT Training')
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset JSON file')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, default='distilbert-base-uncased',
                        help='Pretrained model name')
    parser.add_argument('--hidden_size', type=int, default=768,
                        help='Hidden size')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # 联邦学习参数
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients')
    parser.add_argument('--clients_per_round', type=int, default=5,
                        help='Number of clients per round')
    parser.add_argument('--num_rounds', type=int, default=50,
                        help='Number of federated rounds')
    parser.add_argument('--local_epochs', type=int, default=1,
                        help='Number of local epochs')
    parser.add_argument('--partition_method', type=str, default='label_distribution',
                        choices=['iid', 'label_distribution', 'dirichlet'],
                        help='Data partition method')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Dirichlet alpha parameter')
    parser.add_argument('--personalized', action='store_true',
                        help='Use personalized federated learning')
    parser.add_argument('--use_global_negatives', action='store_true',
                        help='Use global negative sampling')
    parser.add_argument('--global_negative_ratio', type=float, default=0.3,
                        help='Ratio of global negatives')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--loss_alpha', type=float, default=0.8,
                        help='Loss alpha parameter (weight for CE loss)')
    parser.add_argument('--loss_m', type=int, default=50,
                        help='Masked BCE m parameter')
    parser.add_argument('--doc_max_length', type=int, default=256,
                        help='Max document length')
    parser.add_argument('--label_max_length', type=int, default=16,
                        help='Max label length')
    
    # 其他参数
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save_model', action='store_true',
                        help='Save model checkpoints')
    
    return parser.parse_args()


def set_seed(seed: int):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def main():
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 保存配置
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("=" * 80)
    print("Federated CTN-LT Training")
    print("=" * 80)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print("=" * 80)
    
    # 加载数据
    print("\nLoading dataset...")
    documents, labels, label_texts = load_dataset_from_json(args.data_path)
    print(f"  Total documents: {len(documents)}")
    print(f"  Total labels: {len(label_texts)}")
    
    # 计算标签频率
    label_frequencies = [0] * len(label_texts)
    for label_list in labels:
        for label_id in label_list:
            label_frequencies[label_id] += 1
    
    print(f"  Label frequency stats:")
    print(f"    Min: {min(label_frequencies)}")
    print(f"    Max: {max(label_frequencies)}")
    print(f"    Mean: {np.mean(label_frequencies):.2f}")
    print(f"    Median: {np.median(label_frequencies):.2f}")
    
    # 创建tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # 创建联邦数据分区
    print(f"\nPartitioning data ({args.partition_method})...")
    partitioner = FederatedDataPartitioner(
        documents=documents,
        labels=labels,
        label_texts=label_texts,
        num_clients=args.num_clients,
        partition_method=args.partition_method,
        alpha=args.alpha
    )
    
    # 打印分区统计
    stats = partitioner.get_statistics()
    print(f"  Client sizes: {stats['client_sizes']}")
    print(f"  Total samples: {stats['total_samples']}")
    
    # 创建客户端数据加载器
    print("\nCreating client dataloaders...")
    client_loaders = partitioner.create_client_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        doc_max_length=args.doc_max_length,
        label_max_length=args.label_max_length
    )
    
    # 创建全局模型
    print("\nInitializing global model...")
    global_model = CTN_LT(
        model_name=args.model_name,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    )
    
    # 创建服务器
    server = FederatedServer(
        global_model=global_model,
        device=args.device
    )
    
    # 更新全局标签池
    print("Updating global label pool...")
    server.update_global_label_pool(
        label_texts=label_texts,
        tokenizer=tokenizer,
        max_length=args.label_max_length
    )
    
    # 创建客户端
    print(f"\nCreating {args.num_clients} clients...")
    clients = []
    for client_id in range(args.num_clients):
        train_loader, val_loader = client_loaders[client_id]
        
        # 为每个客户端创建模型副本
        client_model = CTN_LT(
            model_name=args.model_name,
            hidden_size=args.hidden_size,
            dropout=args.dropout
        )
        
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=args.device,
            learning_rate=args.learning_rate,
            alpha=args.loss_alpha,
            m=args.loss_m,
            personalized=args.personalized
        )
        
        clients.append(client)
        print(f"  Client {client_id}: {len(train_loader)} train batches")
    
    # 创建联邦训练器
    print("\nInitializing federated trainer...")
    trainer = FederatedTrainer(
        server=server,
        clients=clients,
        num_rounds=args.num_rounds,
        clients_per_round=args.clients_per_round,
        local_epochs=args.local_epochs,
        use_global_negatives=args.use_global_negatives,
        global_negative_ratio=args.global_negative_ratio,
        personalized=args.personalized
    )
    
    # 开始训练
    print("\n" + "=" * 80)
    print("Starting Federated Training")
    print("=" * 80)
    
    history = trainer.train()
    
    # 保存训练历史
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to {history_path}")
    
    # 保存全局模型
    if args.save_model:
        model_path = os.path.join(args.output_dir, 'global_model.pt')
        torch.save(server.global_model.state_dict(), model_path)
        print(f"Global model saved to {model_path}")
        
        # 保存每个客户端的模型
        for client in clients:
            client_model_path = os.path.join(
                args.output_dir, f'client_{client.client_id}_model.pt'
            )
            torch.save(client.model.state_dict(), client_model_path)
        print(f"Client models saved to {args.output_dir}")
    
    print("\n" + "=" * 80)
    print("Training Completed!")
    print("=" * 80)
    
    # 打印最终结果
    print(f"\nFinal Results:")
    print(f"  Average Loss: {history['avg_loss'][-1]:.4f}")
    print(f"  Average CE Loss: {history['avg_ce_loss'][-1]:.4f}")
    print(f"  Average mBCE Loss: {history['avg_mbce_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
