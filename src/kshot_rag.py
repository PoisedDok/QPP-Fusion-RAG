"""
K-Shot RAG Experiment Operation.

Systematic evaluation of RAG performance across different shot counts.
Iterates over k values (e.g., 0, 1, 2, 3, 4, 5, 6, 10) and for each:
1. Builds context from top-k documents
2. Generates answer with LLM
3. Computes QA metrics (Exact Match, F1) against ground truth

Part of QPP-Fusion RAG workflow for ECIR paper reproduction.
"""

import time
import re
from typing import Dict, List, Any, Optional
from collections import Counter


class KShotRAGExperimentOperation:
    """Execute k-shot RAG experiment with QA evaluation."""
    
    def __init__(self, executor=None):
        self.executor = executor
    
    def execute(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        ground_truth: Optional[str] = None,
        shot_counts: str = "0,1,2,3,4,5,6,10",
        model: str = "qwen/qwen3-4b-2507",
        temperature: float = 0.1,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute k-shot RAG experiment.
        
        Args:
            query: Question to answer
            documents: Ranked list of retrieved documents
            ground_truth: Expected answer for evaluation
            shot_counts: Comma-separated k values to evaluate
            model: LLM model name
            temperature: Generation temperature
            max_tokens: Max tokens per response
            system_prompt: System prompt template
            
        Returns:
            Dict with per-shot results, best settings, and summary
        """
        start_time = time.time()
        
        try:
            # Parse shot counts
            k_values = [int(k.strip()) for k in shot_counts.split(",")]
            
            # Default system prompt
            if not system_prompt:
                system_prompt = (
                    "You are a precise question answering assistant. "
                    "Answer the question using ONLY the provided context. "
                    "If the answer is not in the context, say 'I cannot answer.' "
                    "Be concise and direct."
                )
            
            results = []
            best_f1 = -1.0
            best_k = 0
            best_answer = ""
            
            for k in k_values:
                # Build context from top-k documents
                context_docs = documents[:k] if k > 0 else []
                context = self._build_context(context_docs)
                
                # Generate answer
                answer = self._generate_answer(
                    query=query,
                    context=context,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    system_prompt=system_prompt,
                )
                
                # Compute metrics if ground truth provided
                em_score = 0.0
                f1_score = 0.0
                if ground_truth:
                    em_score = self._compute_exact_match(answer, ground_truth)
                    f1_score = self._compute_f1(answer, ground_truth)
                
                result = {
                    "k": k,
                    "answer": answer,
                    "exact_match": em_score,
                    "f1": f1_score,
                    "context_length": len(context),
                    "num_docs": len(context_docs),
                }
                results.append(result)
                
                # Track best
                if f1_score > best_f1:
                    best_f1 = f1_score
                    best_k = k
                    best_answer = answer
            
            # Build summary statistics
            summary = self._compute_summary(results)
            
            # Build per-shot outputs
            output = {
                "results": results,
                "best_k": best_k,
                "best_answer": best_answer,
                "best_f1": best_f1,
                "summary": summary,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }
            
            # Add per-shot direct outputs
            for r in results:
                k = r["k"]
                output[f"answer_{k}shot"] = r["answer"]
                output[f"f1_{k}shot"] = r["f1"]
                output[f"em_{k}shot"] = r["exact_match"]
            
            return output
            
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from documents."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            content = doc.get("content", "")
            title = doc.get("meta", {}).get("title", "")
            if title:
                context_parts.append(f"[Document {i}] {title}\n{content}")
            else:
                context_parts.append(f"[Document {i}]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _generate_answer(
        self,
        query: str,
        context: str,
        model: str,
        temperature: float,
        max_tokens: int,
        system_prompt: str,
    ) -> str:
        """Generate answer using LLM."""
        try:
            from .generation import GenerationOperation
            
            # Build prompt
            if context:
                user_prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"
            else:
                user_prompt = f"Question: {query}\n\nAnswer:"
            
            gen_op = GenerationOperation(self.executor)
            result = gen_op.execute(
                prompt=user_prompt,
                system_prompt=system_prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            return result.get("answer", result.get("response", ""))
            
        except Exception as e:
            return f"[Generation Error: {str(e)}]"
    
    def _normalize_answer(self, text: str) -> str:
        """Normalize answer for comparison."""
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
        pred_norm = self._normalize_answer(prediction)
        gt_norm = self._normalize_answer(ground_truth)
        return 1.0 if pred_norm == gt_norm else 0.0
    
    def _compute_f1(self, prediction: str, ground_truth: str) -> float:
        """Compute token-level F1 score."""
        pred_tokens = self._normalize_answer(prediction).split()
        gt_tokens = self._normalize_answer(ground_truth).split()
        
        if not pred_tokens or not gt_tokens:
            return 0.0 if pred_tokens != gt_tokens else 1.0
        
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_common = sum(common.values())
        
        if num_common == 0:
            return 0.0
        
        precision = num_common / len(pred_tokens)
        recall = num_common / len(gt_tokens)
        
        f1 = 2 * precision * recall / (precision + recall)
        return f1
    
    def _compute_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute summary statistics across all shots."""
        if not results:
            return {}
        
        f1_scores = [r["f1"] for r in results]
        em_scores = [r["exact_match"] for r in results]
        
        return {
            "num_shots_evaluated": len(results),
            "mean_f1": sum(f1_scores) / len(f1_scores),
            "max_f1": max(f1_scores),
            "min_f1": min(f1_scores),
            "mean_em": sum(em_scores) / len(em_scores),
            "best_k_for_f1": results[f1_scores.index(max(f1_scores))]["k"],
            "f1_by_k": {r["k"]: r["f1"] for r in results},
            "em_by_k": {r["k"]: r["exact_match"] for r in results},
        }

