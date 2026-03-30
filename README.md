# Federated CTN-LT

联邦学习版本的CTN-LT (Contrastive Transformer Network for Long Tail Classification)

## 项目简介

本项目实现了将CTN-LT与联邦学习相结合的创新方法，用于解决多标签文本分类中的长尾问题，同时保护数据隐私。

### 核心创新点

1. **联邦对比学习框架**: 在联邦环境下实现对比学习，使用混合负采样策略（本地+全局）
2. **个性化双编码器**: 标签编码器全局共享，文档编码器本地个性化
3. **联邦长尾感知损失**: 结合mBCE和改进的CE损失，专门优化尾部标签性能
4. **全局标签池**: 维护全局标签嵌入池，支持跨客户端知识共享

### 主要特性

- ✅ 支持多种数据分区方法（IID、标签分布、Dirichlet）
- ✅ 个性化联邦学习（可选）
- ✅ 全局负样本采样
- ✅ 完整的评估指标（P@k, N@k, PSP@k, PSN@k）
- ✅ 头部/尾部/零样本标签分析
- ✅ 模型检查点保存

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (推荐)

### 安装依赖

```bash
pip install torch transformers numpy scikit-learn
```

或使用requirements.txt:

```bash
pip install -r requirements.txt
```

## 快速开始

### 1. 准备数据

数据格式为JSON文件：

```json
{
  "documents": ["text1", "text2", ...],
  "labels": [[0, 1, 2], [1, 3], ...],
  "label_texts": ["label0", "label1", ...]
}
```

示例数据生成脚本：

```python
import json

# 创建示例数据
data = {
    "documents": [f"This is document {i}" for i in range(1000)],
    "labels": [[i % 10, (i+1) % 10] for i in range(1000)],
    "label_texts": [f"label_{i}" for i in range(10)]
}

with open('data/sample_data.json', 'w') as f:
    json.dump(data, f)
```

### 2. 训练模型

基础训练命令：

```bash
python -m federated_ctn_lt.train \
    --data_path data/sample_data.json \
    --output_dir outputs/experiment1 \
    --num_clients 10 \
    --num_rounds 50 \
    --save_model
```

使用个性化联邦学习：

```bash
python -m federated_ctn_lt.train \
    --data_path data/sample_data.json \
    --output_dir outputs/personalized \
    --num_clients 10 \
    --num_rounds 50 \
    --personalized \
    --use_global_negatives \
    --save_model
```

使用Dirichlet分区（更真实的Non-IID场景）：

```bash
python -m federated_ctn_lt.train \
    --data_path data/sample_data.json \
    --output_dir outputs/dirichlet \
    --num_clients 10 \
    --partition_method dirichlet \
    --alpha 0.5 \
    --num_rounds 50 \
    --save_model
```

### 3. 参数说明

#### 数据参数
- `--data_path`: 数据集JSON文件路径
- `--output_dir`: 输出目录

#### 模型参数
- `--model_name`: 预训练模型名称（默认: distilbert-base-uncased）
- `--hidden_size`: 隐藏层大小（默认: 768）
- `--dropout`: Dropout率（默认: 0.1）

#### 联邦学习参数
- `--num_clients`: 客户端数量（默认: 10）
- `--clients_per_round`: 每轮参与的客户端数（默认: 5）
- `--num_rounds`: 联邦学习轮数（默认: 50）
- `--local_epochs`: 本地训练轮数（默认: 1）
- `--partition_method`: 数据分区方法（iid/label_distribution/dirichlet）
- `--alpha`: Dirichlet分布参数（默认: 0.5）
- `--personalized`: 使用个性化联邦学习
- `--use_global_negatives`: 使用全局负样本
- `--global_negative_ratio`: 全局负样本比例（默认: 0.3）

#### 训练参数
- `--batch_size`: 批次大小（默认: 32）
- `--learning_rate`: 学习率（默认: 5e-5）
- `--loss_alpha`: CE损失权重（默认: 0.8）
- `--loss_m`: mBCE的m参数（默认: 50）
- `--doc_max_length`: 文档最大长度（默认: 256）
- `--label_max_length`: 标签最大长度（默认: 16）

## 项目结构

```
federated_ctn_lt/
├── models/
│   ├── __init__.py
│   └── ctn_lt.py              # CTN-LT模型实现
├── federated/
│   ├── __init__.py
│   └── fed_trainer.py         # 联邦学习框架
├── data/
│   ├── __init__.py
│   └── data_utils.py          # 数据处理工具
├── evaluation/
│   ├── __init__.py
│   └── metrics.py             # 评估指标
├── train.py                   # 主训练脚本
└── README.md
```

## 使用示例

### 示例1: 基础联邦学习

```python
from federated_ctn_lt.models import CTN_LT
from federated_ctn_lt.federated import FederatedServer, FederatedClient, FederatedTrainer
from federated_ctn_lt.data import FederatedDataPartitioner
from transformers import AutoTokenizer

# 加载数据
documents = [...]  # 文档列表
labels = [...]     # 标签列表
label_texts = [...] # 标签文本

# 创建分区器
partitioner = FederatedDataPartitioner(
    documents, labels, label_texts,
    num_clients=10,
    partition_method='label_distribution'
)

# 创建tokenizer
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# 创建数据加载器
client_loaders = partitioner.create_client_dataloaders(tokenizer, batch_size=32)

# 创建全局模型和服务器
global_model = CTN_LT()
server = FederatedServer(global_model, device='cuda')

# 创建客户端
clients = []
for client_id in range(10):
    train_loader, val_loader = client_loaders[client_id]
    client_model = CTN_LT()
    client = FederatedClient(
        client_id, client_model, train_loader, val_loader,
        device='cuda'
    )
    clients.append(client)

# 创建训练器并训练
trainer = FederatedTrainer(server, clients, num_rounds=50)
history = trainer.train()
```

### 示例2: 评估模型

```python
from federated_ctn_lt.evaluation import MetricsCalculator, LabelFrequencyAnalyzer
import torch

# 创建指标计算器
calculator = MetricsCalculator(num_labels=100, label_frequencies=label_freqs)

# 计算指标
predictions = model(...)  # [batch_size, num_labels]
targets = ...             # [batch_size, num_labels]

metrics = calculator.evaluate_all(predictions, targets, k_values=[1, 3, 5])

print(f"P@5: {metrics['P@5']:.4f}")
print(f"N@5: {metrics['N@5']:.4f}")
print(f"PSP@5: {metrics['PSP@5']:.4f}")
print(f"PSN@5: {metrics['PSN@5']:.4f}")

# 按标签频率分析
analyzer = LabelFrequencyAnalyzer(label_frequencies)
freq_results = analyzer.evaluate_by_frequency(predictions, targets, k=5)

print(f"Frequent labels recall: {freq_results['frequent']['recall']:.4f}")
print(f"Few-shot labels recall: {freq_results['few_shot']['recall']:.4f}")
print(f"Zero-shot labels recall: {freq_results['zero_shot']['recall']:.4f}")
```

## 实验结果

### 数据集

我们在以下数据集上进行了实验：

- **EURLex-4K**: 法律文档分类（3,993个标签）
- **Wiki10-31K**: 维基百科标签预测（30,938个标签）
- **AmazonCat-13K**: 电商商品分类（13,330个标签）

### 性能对比

| 方法 | P@5 | N@5 | PSP@5 | PSN@5 |
|------|-----|-----|-------|-------|
| 中心化CTN-LT | 0.XX | 0.XX | 0.XX | 0.XX |
| FedAvg + CTN-LT | 0.XX | 0.XX | 0.XX | 0.XX |
| **Federated CTN-LT (Ours)** | **0.XX** | **0.XX** | **0.XX** | **0.XX** |

*注: 具体数值需要实际实验后填写*

## 引用

如果您使用了本项目的代码，请引用：

```bibtex
@article{melsbach2025ctnlt,
  title={Contrastive Transformer Network for Long Tail Classification},
  author={Melsbach, Johannes and Haase, Frederic and Stahlmann, Sven and Hirschmeier, Stefan and Schoder, Detlef},
  journal={Knowledge-Based Systems},
  volume={320},
  pages={113607},
  year={2025},
  publisher={Elsevier}
}

@misc{federated_ctn_lt,
  title={Federated CTN-LT: Federated Learning for Long-Tail Multi-Label Text Classification},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/federated-ctn-lt}}
}
```

## 相关资源

- **原始CTN-LT论文**: [Knowledge-Based Systems 2025](https://doi.org/10.1016/j.knosys.2025.113607)
- **CTN-LT代码**: [https://github.com/jmelsbach/CTN-LT](https://github.com/jmelsbach/CTN-LT)
- **联邦学习综述**: McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data"

## 许可证

本项目采用MIT许可证。详见LICENSE文件。

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请联系：
- Email: your.email@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

## 致谢

- 感谢CTN-LT原作者提供的优秀工作
- 感谢Hugging Face提供的Transformers库
- 感谢所有贡献者

---

**注意**: 本项目仍在开发中，欢迎反馈和建议！
