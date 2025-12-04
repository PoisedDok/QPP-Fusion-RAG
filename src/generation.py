"""Generation and validation operations."""

import time
import requests
from typing import Dict, List, Any, Optional


class GenerationOperation:
    """
    LLM Generation via LM Studio API (OpenAI-compatible).
    
    LM Studio runs at localhost:1234 with OpenAI-compatible API.
    """
    
    def __init__(self, executor=None):
        self.executor = executor
        self.base_url = "http://localhost:1234/v1"
    
    def execute(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        model: str = "qwen/qwen3-4b-2507",
        temperature: float = 0.1,
        max_tokens: int = 256,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate response via LM Studio API."""
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
                timeout=60
            )
            response.raise_for_status()
            
            data = response.json()
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
        
        except requests.exceptions.ConnectionError:
            return {
                "error": "LM Studio not running at localhost:1234",
                "error_type": "ConnectionError",
                "answer": "",
                "response": "",
            }
        except Exception as e:
            return {
                "error": str(e),
                "error_type": type(e).__name__,
                "answer": "",
                "response": "",
            }


class GenerateOperation:
    """Atomic generation operation (legacy, uses executor)."""
    
    def __init__(self, executor=None):
        self.executor = executor
    
    def execute(
        self,
        prompt: str,
        context: List[Dict[str, Any]] = None,
        model: str = None,
        provider: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute LLM generation."""
        start_time = time.time()
        
        try:
            if context:
                context_text = '\n\n'.join([doc['content'] for doc in context])
                full_prompt = f"Context:\n{context_text}\n\nQuery: {prompt}\n\nAnswer:"
            else:
                full_prompt = prompt
            
            generator = self.executor.get_generator()
            result = generator.run(prompt=full_prompt)
            answer = result.get('replies', [''])[0]
            
            prompt_tokens = len(full_prompt.split()) * 1.3
            completion_tokens = len(answer.split()) * 1.3
            
            processing_time = time.time() - start_time
            
            return {
                'answer': answer,
                'model': model or self.executor.gen_model,
                'tokens_used': int(prompt_tokens + completion_tokens),
                'latency_ms': processing_time * 1000,
                'metadata': {
                    'prompt_tokens': int(prompt_tokens),
                    'completion_tokens': int(completion_tokens),
                    'temperature': temperature
                }
            }
        
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'answer': '',
                'tokens_used': 0
            }


class ValidateOperation:
    """Atomic validation operation."""
    
    def __init__(self, executor=None):
        self.executor = executor
    
    def execute(
        self,
        answer: str,
        query: str,
        context: List[Dict[str, Any]],
        checks: List[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute answer validation."""
        start_time = time.time()
        
        try:
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
        
        except Exception as e:
            return {
                'error': str(e),
                'error_type': type(e).__name__,
                'is_valid': False,
                'confidence': 0.0
            }

