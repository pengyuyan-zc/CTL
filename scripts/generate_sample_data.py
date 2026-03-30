"""
Generate sample dataset for testing Federated CTN-LT
生成示例数据集用于测试
"""

import json
import random
import argparse
import os


def generate_sample_data(
    num_documents: int = 1000,
    num_labels: int = 50,
    min_labels_per_doc: int = 1,
    max_labels_per_doc: int = 5,
    long_tail: bool = True,
    output_path: str = 'data/sample_data.json'
):
    """
    生成示例多标签文本分类数据集
    
    Args:
        num_documents: 文档数量
        num_labels: 标签数量
        min_labels_per_doc: 每个文档最少标签数
        max_labels_per_doc: 每个文档最多标签数
        long_tail: 是否生成长尾分布
        output_path: 输出文件路径
    """
    print(f"Generating sample dataset...")
    print(f"  Documents: {num_documents}")
    print(f"  Labels: {num_labels}")
    print(f"  Long-tail: {long_tail}")
    
    # 生成标签文本
    label_texts = [f"label_{i}" for i in range(num_labels)]
    
    # 生成标签频率（长尾分布）
    if long_tail:
        # 使用幂律分布
        label_probs = [1.0 / (i + 1) ** 1.5 for i in range(num_labels)]
        # 归一化
        total_prob = sum(label_probs)
        label_probs = [p / total_prob for p in label_probs]
    else:
        # 均匀分布
        label_probs = [1.0 / num_labels] * num_labels
    
    # 生成文档和标签
    documents = []
    labels = []
    
    for doc_id in range(num_documents):
        # 生成文档文本
        doc_text = f"This is document {doc_id}. "
        
        # 随机选择标签数量
        num_doc_labels = random.randint(min_labels_per_doc, max_labels_per_doc)
        
        # 根据概率选择标签
        if long_tail:
            doc_labels = random.choices(
                range(num_labels),
                weights=label_probs,
                k=num_doc_labels
            )
        else:
            doc_labels = random.sample(range(num_labels), num_doc_labels)
        
        # 去重并排序
        doc_labels = sorted(list(set(doc_labels)))
        
        # 将标签添加到文档文本中
        for label_id in doc_labels:
            doc_text += f"{label_texts[label_id]} "
        
        documents.append(doc_text.strip())
        labels.append(doc_labels)
    
    # 统计标签频率
    label_counts = [0] * num_labels
    for doc_labels in labels:
        for label_id in doc_labels:
            label_counts[label_id] += 1
    
    print(f"\nLabel frequency statistics:")
    print(f"  Min: {min(label_counts)}")
    print(f"  Max: {max(label_counts)}")
    print(f"  Mean: {sum(label_counts) / len(label_counts):.2f}")
    print(f"  Median: {sorted(label_counts)[len(label_counts) // 2]}")
    
    # 统计不同频率范围的标签数量
    frequent = sum(1 for c in label_counts if c > 50)
    few_shot = sum(1 for c in label_counts if 1 <= c <= 50)
    zero_shot = sum(1 for c in label_counts if c == 0)
    
    print(f"\nLabel distribution:")
    print(f"  Frequent (>50): {frequent}")
    print(f"  Few-shot (1-50): {few_shot}")
    print(f"  Zero-shot (0): {zero_shot}")
    
    # 保存数据
    data = {
        'documents': documents,
        'labels': labels,
        'label_texts': label_texts
    }
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"\nDataset saved to {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024:.2f} KB")


def main():
    parser = argparse.ArgumentParser(description='Generate sample dataset')
    
    parser.add_argument('--num_documents', type=int, default=1000,
                        help='Number of documents')
    parser.add_argument('--num_labels', type=int, default=50,
                        help='Number of labels')
    parser.add_argument('--min_labels_per_doc', type=int, default=1,
                        help='Minimum labels per document')
    parser.add_argument('--max_labels_per_doc', type=int, default=5,
                        help='Maximum labels per document')
    parser.add_argument('--long_tail', action='store_true',
                        help='Generate long-tail distribution')
    parser.add_argument('--output_path', type=str, default='data/sample_data.json',
                        help='Output file path')
    
    args = parser.parse_args()
    
    generate_sample_data(
        num_documents=args.num_documents,
        num_labels=args.num_labels,
        min_labels_per_doc=args.min_labels_per_doc,
        max_labels_per_doc=args.max_labels_per_doc,
        long_tail=args.long_tail,
        output_path=args.output_path
    )


if __name__ == "__main__":
    main()
