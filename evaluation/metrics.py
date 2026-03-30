"""
Evaluation metrics for multi-label text classification
实现P@k, N@k, PSP@k, PSN@k等指标
"""

import torch
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict


class MetricsCalculator:
    """
    多标签分类指标计算器
    """
    def __init__(self, num_labels: int, label_frequencies: Optional[List[int]] = None):
        self.num_labels = num_labels
        self.label_frequencies = label_frequencies
        
        # 计算倾向性分数
        if label_frequencies is not None:
            self.propensity_scores = self._calculate_propensity_scores(label_frequencies)
        else:
            self.propensity_scores = None
    
    def _calculate_propensity_scores(self, label_frequencies: List[int]) -> np.ndarray:
        """
        计算倾向性分数
        p_l = (C + 1) / (freq_l + C)
        其中C是平滑参数，通常设为1
        """
        C = 1.0
        frequencies = np.array(label_frequencies, dtype=np.float32)
        propensity = (C + 1) / (frequencies + C)
        return propensity
    
    def precision_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算P@k
        Args:
            predictions: [batch_size, num_labels] - 预测分数
            targets: [batch_size, num_labels] - 真实标签（二值）
            k: top-k
        Returns:
            p_at_k: P@k分数
        """
        batch_size = predictions.size(0)
        
        # 获取top-k预测
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        # 计算精度
        total_precision = 0.0
        for i in range(batch_size):
            # 获取真实标签
            true_labels = targets[i].nonzero(as_tuple=True)[0]
            if len(true_labels) == 0:
                continue
            
            # 获取top-k预测
            pred_labels = top_k_indices[i]
            
            # 计算命中数
            hits = sum(1 for label in pred_labels if label in true_labels)
            total_precision += hits / k
        
        return total_precision / batch_size if batch_size > 0 else 0.0
    
    def ndcg_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算nDCG@k
        Args:
            predictions: [batch_size, num_labels]
            targets: [batch_size, num_labels]
            k: top-k
        Returns:
            ndcg: nDCG@k分数
        """
        batch_size = predictions.size(0)
        
        # 获取top-k预测
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        total_ndcg = 0.0
        for i in range(batch_size):
            # 获取真实标签
            true_labels = targets[i].nonzero(as_tuple=True)[0]
            if len(true_labels) == 0:
                continue
            
            # 计算DCG
            dcg = 0.0
            for rank, label_idx in enumerate(top_k_indices[i]):
                if label_idx in true_labels:
                    dcg += 1.0 / np.log2(rank + 2)  # rank从0开始，所以+2
            
            # 计算IDCG（理想情况下的DCG）
            idcg = sum(1.0 / np.log2(rank + 2) for rank in range(min(k, len(true_labels))))
            
            # 计算nDCG
            if idcg > 0:
                total_ndcg += dcg / idcg
        
        return total_ndcg / batch_size if batch_size > 0 else 0.0
    
    def propensity_scored_precision_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算PSP@k（倾向性评分精度）
        """
        if self.propensity_scores is None:
            raise ValueError("Propensity scores not available. Provide label frequencies.")
        
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        total_psp = 0.0
        for i in range(batch_size):
            true_labels = targets[i].nonzero(as_tuple=True)[0]
            if len(true_labels) == 0:
                continue
            
            pred_labels = top_k_indices[i]
            
            # 计算加权命中
            weighted_hits = 0.0
            for label in pred_labels:
                if label in true_labels:
                    weighted_hits += 1.0 / self.propensity_scores[label.item()]
            
            total_psp += weighted_hits / k
        
        return total_psp / batch_size if batch_size > 0 else 0.0
    
    def propensity_scored_ndcg_at_k(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ) -> float:
        """
        计算PSnDCG@k
        """
        if self.propensity_scores is None:
            raise ValueError("Propensity scores not available.")
        
        batch_size = predictions.size(0)
        _, top_k_indices = torch.topk(predictions, k, dim=1)
        
        total_psndcg = 0.0
        for i in range(batch_size):
            true_labels = targets[i].nonzero(as_tuple=True)[0]
            if len(true_labels) == 0:
                continue
            
            # 计算PSDCG
            psdcg = 0.0
            for rank, label_idx in enumerate(top_k_indices[i]):
                if label_idx in true_labels:
                    psdcg += (1.0 / self.propensity_scores[label_idx.item()]) / np.log2(rank + 2)
            
            # 计算理想PSDCG
            # 按倾向性分数排序真实标签
            true_labels_list = true_labels.tolist()
            true_propensities = [1.0 / self.propensity_scores[l] for l in true_labels_list]
            sorted_propensities = sorted(true_propensities, reverse=True)
            
            ideal_psdcg = sum(
                prop / np.log2(rank + 2)
                for rank, prop in enumerate(sorted_propensities[:k])
            )
            
            if ideal_psdcg > 0:
                total_psndcg += psdcg / ideal_psdcg
        
        return total_psndcg / batch_size if batch_size > 0 else 0.0
    
    def evaluate_all(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k_values: List[int] = [1, 3, 5]
    ) -> Dict[str, float]:
        """
        计算所有指标
        """
        metrics = {}
        
        for k in k_values:
            metrics[f'P@{k}'] = self.precision_at_k(predictions, targets, k)
            metrics[f'N@{k}'] = self.ndcg_at_k(predictions, targets, k)
            
            if self.propensity_scores is not None:
                metrics[f'PSP@{k}'] = self.propensity_scored_precision_at_k(predictions, targets, k)
                metrics[f'PSN@{k}'] = self.propensity_scored_ndcg_at_k(predictions, targets, k)
        
        return metrics


class LabelFrequencyAnalyzer:
    """
    标签频率分析器
    用于分析头部、尾部、零样本标签的性能
    """
    def __init__(
        self,
        label_frequencies: List[int],
        frequent_threshold: int = 50,
        few_shot_threshold: int = 1
    ):
        self.label_frequencies = np.array(label_frequencies)
        self.frequent_threshold = frequent_threshold
        self.few_shot_threshold = few_shot_threshold
        
        # 分类标签
        self.frequent_labels = np.where(self.label_frequencies > frequent_threshold)[0]
        self.few_shot_labels = np.where(
            (self.label_frequencies >= few_shot_threshold) &
            (self.label_frequencies <= frequent_threshold)
        )[0]
        self.zero_shot_labels = np.where(self.label_frequencies == 0)[0]
    
    def evaluate_by_frequency(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        按标签频率分组评估
        """
        results = {}
        
        # 评估头部标签
        if len(self.frequent_labels) > 0:
            freq_metrics = self._evaluate_label_subset(
                predictions, targets, self.frequent_labels, k
            )
            results['frequent'] = freq_metrics
        
        # 评估few-shot标签
        if len(self.few_shot_labels) > 0:
            few_shot_metrics = self._evaluate_label_subset(
                predictions, targets, self.few_shot_labels, k
            )
            results['few_shot'] = few_shot_metrics
        
        # 评估zero-shot标签
        if len(self.zero_shot_labels) > 0:
            zero_shot_metrics = self._evaluate_label_subset(
                predictions, targets, self.zero_shot_labels, k
            )
            results['zero_shot'] = zero_shot_metrics
        
        return results
    
    def _evaluate_label_subset(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        label_indices: np.ndarray,
        k: int
    ) -> Dict[str, float]:
        """
        评估特定标签子集
        """
        # 创建掩码
        mask = torch.zeros_like(predictions, dtype=torch.bool)
        mask[:, label_indices] = True
        
        # 过滤预测和目标
        masked_predictions = predictions.clone()
        masked_predictions[~mask] = float('-inf')
        
        masked_targets = targets.clone()
        masked_targets[~mask] = 0
        
        # 只保留至少有一个目标标签的样本
        valid_samples = masked_targets.sum(dim=1) > 0
        if valid_samples.sum() == 0:
            return {'recall': 0.0, 'precision': 0.0}
        
        masked_predictions = masked_predictions[valid_samples]
        masked_targets = masked_targets[valid_samples]
        
        # 计算指标
        _, top_k_indices = torch.topk(masked_predictions, min(k, len(label_indices)), dim=1)
        
        total_recall = 0.0
        total_precision = 0.0
        num_samples = masked_predictions.size(0)
        
        for i in range(num_samples):
            true_labels = masked_targets[i].nonzero(as_tuple=True)[0]
            pred_labels = top_k_indices[i]
            
            hits = sum(1 for label in pred_labels if label in true_labels)
            
            if len(true_labels) > 0:
                total_recall += hits / len(true_labels)
            total_precision += hits / len(pred_labels)
        
        return {
            'recall': total_recall / num_samples,
            'precision': total_precision / num_samples
        }


if __name__ == "__main__":
    # 测试代码
    print("Testing evaluation metrics...")
    
    # 创建虚拟数据
    batch_size = 10
    num_labels = 20
    
    predictions = torch.randn(batch_size, num_labels)
    targets = torch.zeros(batch_size, num_labels)
    
    # 随机设置一些真实标签
    for i in range(batch_size):
        num_true_labels = np.random.randint(1, 5)
        true_label_indices = np.random.choice(num_labels, num_true_labels, replace=False)
        targets[i, true_label_indices] = 1
    
    # 创建标签频率
    label_frequencies = [100, 80, 60, 50, 40, 30, 20, 10, 5, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    
    # 测试指标计算器
    calculator = MetricsCalculator(num_labels, label_frequencies)
    
    metrics = calculator.evaluate_all(predictions, targets, k_values=[1, 3, 5])
    
    print("Metrics:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")
    
    # 测试频率分析器
    analyzer = LabelFrequencyAnalyzer(label_frequencies)
    
    freq_results = analyzer.evaluate_by_frequency(predictions, targets, k=5)
    
    print("\nFrequency-based evaluation:")
    for category, metrics in freq_results.items():
        print(f"  {category}:")
        for metric_name, value in metrics.items():
            print(f"    {metric_name}: {value:.4f}")
    
    print("\nEvaluation test passed!")
