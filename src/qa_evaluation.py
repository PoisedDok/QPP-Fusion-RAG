"""
QA Evaluation Operation.

Computes question answering metrics:
- Exact Match (EM): Binary score, 1 if normalized strings match exactly
- Token F1: Token-level overlap score
- ROUGE-L: Longest common subsequence based metric

Used to evaluate RAG system answer quality against ground truth.
"""

import re
import time
from typing import Dict, List, Any, Optional
from collections import Counter


class QAEvaluationOperation:
    """Execute QA evaluation metrics computation."""
    
    def __init__(self, executor=None):
        self.executor = executor
    
    def execute(
        self,
        prediction: str,
        ground_truth: str,
        metrics: Optional[List[str]] = None,
        normalize: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute QA evaluation metrics.
        
        Args:
            prediction: Predicted answer from RAG system
            ground_truth: Expected correct answer
            metrics: List of metrics to compute ['exact_match', 'f1', 'rouge_l']
            normalize: Whether to normalize answers before comparison
            
        Returns:
            Dict with metric scores and pass/fail status
        """
        start_time = time.time()
        
        try:
            # Default metrics
            if not metrics:
                metrics = ['exact_match', 'f1']
            
            # Normalize if requested
            if normalize:
                pred_norm = self._normalize_answer(prediction)
                gt_norm = self._normalize_answer(ground_truth)
            else:
                pred_norm = prediction.strip()
                gt_norm = ground_truth.strip()
            
            results = {}
            
            # Compute requested metrics
            if 'exact_match' in metrics:
                results['exact_match'] = self._compute_exact_match(pred_norm, gt_norm)
            
            if 'f1' in metrics:
                results['f1'] = self._compute_f1(pred_norm, gt_norm)
            
            if 'rouge_l' in metrics:
                results['rouge_l'] = self._compute_rouge_l(pred_norm, gt_norm)
            
            # Determine if passed (EM=1 or F1>0.5)
            em = results.get('exact_match', 0)
            f1 = results.get('f1', 0)
            passed = em == 1.0 or f1 > 0.5
            
            return {
                **results,
                'all_metrics': results,
                'passed': passed,
                'prediction_normalized': pred_norm,
                'ground_truth_normalized': gt_norm,
                'processing_time_ms': (time.time() - start_time) * 1000,
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'processing_time_ms': (time.time() - start_time) * 1000,
            }
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison (SQuAD style)."""
        # Lowercase
        text = text.lower()
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Remove articles
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text.strip()
    
    def _compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        """Compute exact match score."""
        return 1.0 if prediction == ground_truth else 0.0
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = prediction.split()
        gt_tokens = ground_truth.split()
        
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _compute_rouge_l(self, prediction: str, ground_truth: str) -> float:
        """Compute ROUGE-L (Longest Common Subsequence) score."""
        pred_tokens = prediction.split()
        gt_tokens = ground_truth.split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0 if pred_tokens or gt_tokens else 1.0
        
        # Compute LCS length
        lcs_length = self._lcs_length(pred_tokens, gt_tokens)
        
        if lcs_length == 0:
            return 0.0
        
        precision = lcs_length / len(pred_tokens)
        recall = lcs_length / len(gt_tokens)
        
        # F-measure
        if precision + recall == 0:
            return 0.0
        
        f_lcs = 2 * precision * recall / (precision + recall)
        return f_lcs
    
    def _lcs_length(self, x: List[str], y: List[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        
        # DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        return dp[m][n]

