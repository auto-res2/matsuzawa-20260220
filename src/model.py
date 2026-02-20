"""LLM model wrapper for inference."""

import os
from typing import Dict, List, Optional
from openai import OpenAI


class LLMModel:
    """Wrapper for LLM inference via API."""
    
    def __init__(
        self,
        model_name: str,
        provider: str = "openai",
        api_key_env: str = "OPENAI_API_KEY",
        base_model: Optional[str] = None,
    ):
        """Initialize LLM model.
        
        Args:
            model_name: Model identifier
            provider: API provider (currently only 'openai' supported)
            api_key_env: Environment variable containing API key
            base_model: Specific model version to use
        """
        self.model_name = model_name
        self.provider = provider
        self.base_model = base_model or model_name
        
        if provider == "openai":
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"API key not found in environment variable: {api_key_env}")
            self.client = OpenAI(api_key=api_key)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[str]:
        """Generate text completion(s).
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions to generate
            
        Returns:
            List of generated completions
        """
        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.base_model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice.message.content for choice in response.choices]
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def generate_batch(
        self,
        prompts: List[str],
        temperature: float = 0.7,
        max_tokens: int = 512,
        n: int = 1,
    ) -> List[List[str]]:
        """Generate completions for a batch of prompts.
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            n: Number of completions per prompt
            
        Returns:
            List of lists of generated completions
        """
        results = []
        for prompt in prompts:
            completions = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                n=n,
            )
            results.append(completions)
        return results


def build_cot_prompt(question: str, method: str = "standard") -> str:
    """Build a chain-of-thought prompt for a math problem.
    
    Args:
        question: The math problem
        method: Prompting method ('standard' or 'proposed')
        
    Returns:
        Formatted prompt string
    """
    # [VALIDATOR FIX - Attempt 1]
    # [PROBLEM]: Low accuracy (14.5%) - answers not being extracted correctly from model outputs
    # [CAUSE]: Prompt doesn't specify a clear format for the final answer, leading to inconsistent formats
    #          that the extraction regex fails to parse
    # [FIX]: Updated prompt to explicitly request the answer in GSM8K standard format (#### <answer>)
    #
    # [OLD CODE]:
    # prompt = f"""Solve the following math problem step by step. Show your reasoning clearly.
    #
    # Problem: {question}
    #
    # Let's solve this step by step:"""
    #
    # [NEW CODE]:
    prompt = f"""Solve the following math problem step by step. Show your reasoning clearly.

Problem: {question}

Let's solve this step by step, and provide the final numeric answer at the end in the format:
#### <final answer>

Solution:"""
    
    return prompt


def build_verification_prompt(
    question: str,
    proposed_answer: str,
    sample_rationales: List[str],
) -> str:
    """Build a falsification verification prompt.
    
    This prompt asks the model to attempt to DISPROVE the proposed answer
    using explicit sanity checks.
    
    Args:
        question: Original math problem
        proposed_answer: Proposed numeric answer to verify
        sample_rationales: Sample rationales that led to this answer
        
    Returns:
        Verification prompt
    """
    rationales_text = "\n\n".join([
        f"Reasoning {i+1}:\n{r}"
        for i, r in enumerate(sample_rationales[:2])  # Limit to 2 for brevity
    ])
    
    prompt = f"""You are a careful verifier. Your task is to attempt to DISPROVE a proposed answer by performing sanity checks.

Problem: {question}

Proposed Answer: {proposed_answer}

Sample reasoning that led to this answer:
{rationales_text}

Perform three independent sanity checks to attempt to falsify this answer:

1. ESTIMATION CHECK: Use rough estimation to verify the answer is in the right ballpark.
2. ALTERNATIVE METHOD: Try to solve using a different approach and check if you get the same result.
3. CONSTRAINT CHECK: Verify the answer satisfies any constraints in the problem (e.g., positive/negative, integer, magnitude).

For each check, output:
- CHECK [1/2/3]: [Brief description]
- RESULT: [PASS/FAIL]

After all checks, provide:
- CHECKS PASSED: [0-3]
- CONFIDENCE: [0.0-1.0] (your confidence the proposed answer is correct)
- VERDICT: [CORRECT/INCORRECT]

Be critical and thorough. If any check fails, explain why."""
    
    return prompt


def parse_verification_response(response: str) -> Dict[str, float]:
    """Parse verification response into structured output.
    
    Args:
        response: Raw verification response text
        
    Returns:
        Dict with 'confidence' and 'checks_passed' keys
    """
    import re
    
    # Extract checks passed
    checks_match = re.search(r"CHECKS PASSED:\s*(\d+)", response, re.IGNORECASE)
    checks_passed = int(checks_match.group(1)) if checks_match else 0
    
    # Extract confidence
    conf_match = re.search(r"CONFIDENCE:\s*(0?\.\d+|1\.0|0|1)", response, re.IGNORECASE)
    confidence = float(conf_match.group(1)) if conf_match else 0.5
    
    # Ensure valid ranges
    checks_passed = max(0, min(3, checks_passed))
    confidence = max(0.0, min(1.0, confidence))
    
    return {
        "confidence": confidence,
        "checks_passed": checks_passed,
    }
