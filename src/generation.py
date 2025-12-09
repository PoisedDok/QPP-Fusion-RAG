"""
Incoming: prompt, context --- {str, List[Dict]}
Processing: LLM generation --- {1 job: API call}
Outgoing: generated answer --- {Dict with answer, metadata}

Generation Operations
---------------------
LLM generation via LM Studio API (OpenAI-compatible).

STRICT: No silent error handling. Connection failures raise exceptions.
"""

import time
import requests
from typing import Dict, List, Any, Optional

# Import config
from src.config import config


class LMStudioConnectionError(Exception):
    """Raised when LM Studio is not running or unreachable."""
    pass


class GenerationError(Exception):
    """Raised when generation fails."""
    pass


class GenerationOperation:
    """
    LLM Generation via LM Studio API (OpenAI-compatible).
    
    LM Studio runs at localhost:1234 with OpenAI-compatible API.
    
    STRICT: Connection failures raise LMStudioConnectionError.
    """
    
    def __init__(self, base_url: str = None):
        """
        Initialize generation operation.
        
        Args:
            base_url: LM Studio API base URL (from config if None)
        """
        self.base_url = base_url or config.models.lm_studio.base_url
        self.default_model = config.models.lm_studio.default_model
        self.timeout = config.models.lm_studio.timeout_seconds
    
    def execute(
        self,
        prompt: str,
        system_prompt: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate response via LM Studio API.
        
        Args:
            prompt: User prompt
            system_prompt: System instructions (from config if None)
            model: Model identifier (from config if None)
            temperature: Sampling temperature (from config if None)
            max_tokens: Maximum tokens to generate (from config if None)
            
        Returns:
            Dict with answer, model, tokens_used, latency_ms, metadata
            
        Raises:
            LMStudioConnectionError: If LM Studio is not running.
            GenerationError: If API call fails.
        """
        # Get defaults from config
        system_prompt = system_prompt or config.generation.system_prompt
        model = model or self.default_model
        temperature = temperature if temperature is not None else config.generation.temperature
        max_tokens = max_tokens or config.generation.max_tokens
        
        start_time = time.time()
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as e:
            raise LMStudioConnectionError(
                f"LM Studio not running at {self.base_url}. "
                f"Start LM Studio and load a model first. Error: {e}"
            )
        except requests.exceptions.Timeout:
            raise GenerationError(f"LM Studio request timed out after {self.timeout}s")
        
        if response.status_code != 200:
            raise GenerationError(
                f"LM Studio API error {response.status_code}: {response.text}"
            )
        
        try:
            data = response.json()
        except ValueError as e:
            raise GenerationError(f"LM Studio returned invalid JSON: {e}")
        
        if "choices" not in data or not data["choices"]:
            raise GenerationError(f"LM Studio response missing choices: {data}")
        
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        
        return {
            "answer": answer,
            "response": answer,
            "model": model,
            "tokens_used": usage.get("total_tokens", 0),
            "latency_ms": (time.time() - start_time) * 1000,
            "metadata": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "temperature": temperature
            }
        }


class ValidateOperation:
    """
    Atomic validation operation.
    
    Validates generated answers against context.
    """
    
    def __init__(self):
        pass
    
    def execute(
        self,
        answer: str,
        query: str,
        context: List[Dict[str, Any]],
        checks: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute answer validation.
        
        Args:
            answer: Generated answer to validate
            query: Original query
            context: Retrieved documents with 'content' field
            checks: Validation checks to run (hallucination, completeness, citation)
            
        Returns:
            Dict with is_valid, checks_passed, checks_failed, confidence, issues
            
        Raises:
            ValueError: If answer or context is empty.
        """
        start_time = time.time()
        
        if not answer:
            raise ValueError("Cannot validate empty answer")
        if not context:
            raise ValueError("Cannot validate without context")
        
        checks = checks or ['hallucination', 'completeness']
        
        checks_passed = []
        checks_failed = []
        issues = []
        
        if 'hallucination' in checks:
            answer_lower = answer.lower()
            context_text = ' '.join([doc['content'].lower() for doc in context])
            
            claims = [s.strip() for s in answer.split('.') if s.strip()]
            hallucinated = False
            
            for claim in claims:
                if len(claim.split()) > 3:
                    if not any(word in context_text for word in claim.lower().split()[:3]):
                        hallucinated = True
                        issues.append(f"Unsupported claim: {claim[:50]}...")
            
            if hallucinated:
                checks_failed.append('hallucination')
            else:
                checks_passed.append('hallucination')
        
        if 'completeness' in checks:
            query_keywords = set(query.lower().split())
            answer_keywords = set(answer.lower().split())
            
            overlap = query_keywords & answer_keywords
            if len(overlap) / len(query_keywords) > 0.3:
                checks_passed.append('completeness')
            else:
                checks_failed.append('completeness')
                issues.append("Answer may not fully address query")
        
        if 'citation' in checks:
            if any(marker in answer for marker in ['[', 'source:', 'according to']):
                checks_passed.append('citation')
            else:
                checks_failed.append('citation')
                issues.append("Missing citations")
        
        processing_time = time.time() - start_time
        is_valid = len(checks_failed) == 0
        confidence = len(checks_passed) / len(checks) if checks else 1.0
        
        return {
            'is_valid': is_valid,
            'checks_passed': checks_passed,
            'checks_failed': checks_failed,
            'confidence': round(confidence, 3),
            'issues': issues,
            'processing_time_ms': processing_time * 1000
        }
