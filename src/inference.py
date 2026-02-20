"""Inference logic for Chain-of-Thought reasoning with Self-Consistency."""

import re
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.model import LLMModel, build_cot_prompt, build_verification_prompt, parse_verification_response
from src.preprocess import load_math_dataset, extract_answer, compare_answers


def extract_final_answer(rationale: str) -> str:
    """Extract the final numeric answer from a rationale.
    
    Args:
        rationale: Chain-of-thought reasoning text
        
    Returns:
        Extracted numeric answer
    """
    # [VALIDATOR FIX - Attempt 3]
    # [PROBLEM]: Low accuracy (12.5%) - extracting wrong numbers from question text or intermediate steps
    # [CAUSE]: Previous patterns were too permissive, matching numbers from problem statement.
    #          Also, the last fallback pattern matched ANY number anywhere in text.
    #          GPT-4o-mini responses don't always use "####" or "**" markers as instructed.
    # [FIX]: 1) Made patterns more restrictive to only match numbers in answer context
    #        2) Added pattern to find last number in the LAST sentence (after final period)
    #        3) Removed overly permissive pattern that matched any number
    #        4) Better handling of numbers with commas and dollar signs
    #
    # [OLD CODE]:
    # patterns = [
    #     r"\*\*[^\d]*?(-?\d+(?:,\d{3})*(?:\.\d+)?)[^\*]*?\*\*",
    #     r"\\boxed\{(-?\d+(?:,\d{3})*(?:\.\d+)?)\}",
    #     r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    #     r"(?:final answer|answer is|result is|solution is)[:\s]+\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
    #     r"(?:the answer is|answer:)\s+\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?:\s+|\.|\*|$)",
    #     r"=\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\.|$)",
    #     r"(?:^|\s)\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\.|$)",
    # ]
    #
    # [NEW CODE]:
    # Look for common answer patterns (ordered from most specific to least specific)
    patterns = [
        # GSM8K format with ####
        r"####\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        # Markdown bold with answer context
        r"\*\*\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\*\*",
        # LaTeX boxed format
        r"\\boxed\{\s*\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*\}",
        # Explicit answer phrases (case insensitive)
        r"(?:final answer|the answer|answer is|result is|solution is|therefore)[:\s]+\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)",
        # "makes/earns/costs/pays X dollars/money"
        r"(?:makes?|earns?|costs?|pays?|totals?|gets?)\s+\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s+(?:dollars?|money)",
        # Dollar amount at end of sentence
        r"\$\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*(?:\.|$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, rationale, re.IGNORECASE)
        if match:
            answer = match.group(1).replace(",", "")
            return answer
    
    # Last resort: find ALL numbers and take the LAST one (likely the final answer)
    # But only from the last 200 characters to avoid grabbing from problem statement
    last_segment = rationale[-200:] if len(rationale) > 200 else rationale
    all_numbers = re.findall(r"(?<!\d)\$?\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)(?!\d)", last_segment)
    if all_numbers:
        return all_numbers[-1].replace(",", "").replace("$", "").strip()
    
    # Final fallback: return the rationale itself
    return rationale.strip()


def sample_cot_chains(
    model: LLMModel,
    question: str,
    num_samples: int,
    temperature: float,
    max_tokens: int,
) -> List[Tuple[str, str]]:
    """Sample multiple Chain-of-Thought solutions.
    
    Args:
        model: LLM model instance
        question: Math problem question
        num_samples: Number of samples to generate
        temperature: Sampling temperature
        max_tokens: Max tokens per generation
        
    Returns:
        List of (rationale, extracted_answer) tuples
    """
    prompt = build_cot_prompt(question)
    
    # Generate samples
    completions = model.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        n=num_samples,
    )
    
    # Extract answers from rationales
    samples = []
    for rationale in completions:
        answer = extract_final_answer(rationale)
        samples.append((rationale, answer))
    
    return samples


def verify_answer_hypothesis(
    model: LLMModel,
    question: str,
    proposed_answer: str,
    sample_rationales: List[str],
) -> Dict[str, float]:
    """Run falsification verification on an answer hypothesis.
    
    Args:
        model: LLM model instance
        question: Original problem
        proposed_answer: Answer to verify
        sample_rationales: Sample rationales that led to this answer
        
    Returns:
        Dict with 'confidence' and 'checks_passed' scores
    """
    prompt = build_verification_prompt(
        question=question,
        proposed_answer=proposed_answer,
        sample_rationales=sample_rationales,
    )
    
    # Generate verification with low temperature for consistency
    response = model.generate(
        prompt=prompt,
        temperature=0.0,
        max_tokens=1024,
        n=1,
    )[0]
    
    # Parse structured output
    verification = parse_verification_response(response)
    
    return verification


def aggregate_with_sfw_sc(
    answer_hypotheses: Dict[str, List[str]],
    verification_results: Dict[str, Dict],
    alpha: float,
    beta: float,
    gamma: float,
) -> str:
    """Aggregate using Selective Falsification-Weighted Self-Consistency.
    
    Args:
        answer_hypotheses: Map from answer to list of rationales
        verification_results: Map from answer to verification dict
        alpha: Frequency weight exponent
        beta: Confidence weight exponent
        gamma: Checks-passed weight exponent
        
    Returns:
        Best answer according to the scoring function
    """
    total_samples = sum(len(rationales) for rationales in answer_hypotheses.values())
    scores = {}
    
    for answer, rationales in answer_hypotheses.items():
        # Frequency term
        frequency = len(rationales) / total_samples
        freq_term = frequency ** alpha
        
        # Verification terms (if available)
        if answer in verification_results:
            verification = verification_results[answer]
            confidence = verification["confidence"]
            checks_passed = verification["checks_passed"] / 3.0  # Normalize to [0, 1]
            
            conf_term = confidence ** beta
            checks_term = checks_passed ** gamma
        else:
            # No verification (baseline mode)
            conf_term = 1.0
            checks_term = 1.0
        
        # Combined score
        scores[answer] = freq_term * conf_term * checks_term
    
    # Return answer with highest score
    best_answer = max(scores.items(), key=lambda x: x[1])[0]
    return best_answer


def run_inference(cfg: DictConfig) -> None:
    """Run inference experiment.
    
    Args:
        cfg: Hydra configuration
    """
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        # Determine project name based on mode
        project = cfg.wandb.project
        if cfg.mode == "sanity_check":
            project = f"{project}-sanity"
        
        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
        print(f"WandB run URL: {wandb.run.url}")
    
    # Load dataset
    print(f"Loading dataset: {cfg.run.dataset.name}")
    max_samples = cfg.run.dataset.max_samples
    
    # In sanity_check mode, reduce to a small subset
    if cfg.mode == "sanity_check":
        max_samples = 10
    
    problems = load_math_dataset(
        dataset_name=cfg.run.dataset.name,
        split=cfg.run.dataset.split,
        cache_dir=cfg.run.inference.cache_dir,
        max_samples=max_samples,
        subset_name=cfg.run.dataset.get("subset_name"),
    )
    
    print(f"Loaded {len(problems)} problems")
    
    # Initialize model
    print(f"Initializing model: {cfg.run.model.name}")
    model = LLMModel(
        model_name=cfg.run.model.name,
        provider=cfg.run.model.provider,
        api_key_env=cfg.run.model.api_key_env,
        base_model=cfg.run.model.base_model,
    )
    
    # Get method config
    method_cfg = cfg.run.method
    verification_enabled = method_cfg.verification.enabled
    
    # Track metrics
    num_correct = 0
    num_total = 0
    all_predictions = []
    
    # Process each problem
    print(f"\nRunning inference with method: {method_cfg.name}")
    for problem in tqdm(problems, desc="Inference"):
        question = problem["question"]
        ground_truth = problem["answer"]
        
        # Step 1: Sample K CoT chains
        samples = sample_cot_chains(
            model=model,
            question=question,
            num_samples=method_cfg.sampling.num_samples,
            temperature=method_cfg.sampling.temperature,
            max_tokens=method_cfg.sampling.max_tokens,
        )
        
        # Step 2: Group by answer hypothesis
        answer_hypotheses = defaultdict(list)
        for rationale, answer in samples:
            answer_hypotheses[answer].append(rationale)
        
        # Step 3: Run verification (if enabled)
        verification_results = {}
        if verification_enabled:
            max_rationales = method_cfg.verification.max_rationales_per_hypothesis
            
            for answer, rationales in answer_hypotheses.items():
                # Select shortest rationales for efficiency
                sorted_rationales = sorted(rationales, key=len)
                sample_rationales = sorted_rationales[:max_rationales]
                
                verification = verify_answer_hypothesis(
                    model=model,
                    question=question,
                    proposed_answer=answer,
                    sample_rationales=sample_rationales,
                )
                verification_results[answer] = verification
        
        # Step 4: Aggregate
        prediction = aggregate_with_sfw_sc(
            answer_hypotheses=answer_hypotheses,
            verification_results=verification_results,
            alpha=method_cfg.aggregation.alpha,
            beta=method_cfg.aggregation.beta,
            gamma=method_cfg.aggregation.gamma,
        )
        
        # Check correctness
        correct = compare_answers(prediction, ground_truth)
        num_correct += int(correct)
        num_total += 1
        
        all_predictions.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "correct": correct,
            "num_hypotheses": len(answer_hypotheses),
        })
        
        # Log to WandB
        if cfg.wandb.mode != "disabled":
            wandb.log({
                "accuracy": num_correct / num_total,
                "num_correct": num_correct,
                "num_total": num_total,
            })
    
    # Calculate final metrics
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    
    print(f"\nFinal Accuracy: {accuracy:.4f} ({num_correct}/{num_total})")
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Save predictions
    predictions_file = results_dir / "predictions.json"
    with open(predictions_file, "w") as f:
        json.dump(all_predictions, f, indent=2)
    
    # Save metrics
    metrics = {
        "accuracy": accuracy,
        "num_correct": num_correct,
        "num_total": num_total,
    }
    
    metrics_file = results_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to: {results_dir}")
    
    # Log final summary to WandB
    if cfg.wandb.mode != "disabled":
        wandb.summary["accuracy"] = accuracy
        wandb.summary["num_correct"] = num_correct
        wandb.summary["num_total"] = num_total
        wandb.finish()
    
    # Sanity validation for sanity_check mode
    if cfg.mode == "sanity_check":
        perform_sanity_validation(
            num_samples=num_total,
            accuracy=accuracy,
        )


def perform_sanity_validation(num_samples: int, accuracy: float) -> None:
    """Perform sanity validation checks.
    
    Args:
        num_samples: Number of samples processed
        accuracy: Final accuracy score
    """
    # Check conditions
    passed = True
    reason = ""
    
    if num_samples < 5:
        passed = False
        reason = "insufficient_samples"
    elif accuracy == 0.0:
        passed = False
        reason = "zero_accuracy"
    elif accuracy > 1.0 or accuracy < 0.0:
        passed = False
        reason = "invalid_accuracy"
    
    # Print summary
    summary = {
        "samples": num_samples,
        "accuracy": accuracy,
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")
    
    # Print verdict
    if passed:
        print("SANITY_VALIDATION: PASS")
    else:
        print(f"SANITY_VALIDATION: FAIL reason={reason}")
        sys.exit(1)
