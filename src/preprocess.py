"""Dataset preprocessing for math reasoning tasks."""

import re
from typing import Dict, List, Optional
from datasets import load_dataset


def load_math_dataset(
    dataset_name: str,
    split: str = "test",
    cache_dir: str = ".cache",
    max_samples: Optional[int] = None,
    subset_name: Optional[str] = None,
) -> List[Dict]:
    """Load math reasoning dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'gsm8k')
        split: Dataset split to load
        cache_dir: Directory for caching datasets
        max_samples: Maximum number of samples to load
        subset_name: Subset name for the dataset
        
    Returns:
        List of problem dictionaries with 'question' and 'answer' keys
    """
    if dataset_name == "gsm8k":
        dataset = load_dataset(
            "openai/gsm8k",
            subset_name or "main",
            split=split,
            cache_dir=cache_dir,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Convert to list of dicts
    problems = []
    for i, item in enumerate(dataset):
        if max_samples and i >= max_samples:
            break
            
        # GSM8K format: question and answer fields
        problems.append({
            "question": item["question"],
            "answer": extract_answer(item["answer"]),
            "full_solution": item["answer"],
        })
    
    return problems


def extract_answer(solution_text: str) -> str:
    """Extract the final numeric answer from a solution.
    
    GSM8K answers are in format: "...#### 42"
    
    Args:
        solution_text: Full solution text
        
    Returns:
        Extracted numeric answer as string
    """
    # GSM8K format: answer after ####
    match = re.search(r"####\s*(-?\d+(?:,\d{3})*(?:\.\d+)?)", solution_text)
    if match:
        # Remove commas from numbers
        answer = match.group(1).replace(",", "")
        return answer
    
    # Fallback: try to find any number at the end
    match = re.search(r"(-?\d+(?:,\d{3})*(?:\.\d+)?)\s*$", solution_text.strip())
    if match:
        return match.group(1).replace(",", "")
    
    return solution_text.strip()


def normalize_numeric_answer(answer: str) -> str:
    """Normalize a numeric answer for comparison.
    
    Args:
        answer: Answer string (may contain commas, dollar signs, etc.)
        
    Returns:
        Normalized numeric string
    """
    # Remove common non-numeric characters
    answer = answer.strip()
    answer = answer.replace("$", "").replace(",", "").replace(" ", "")
    
    # Try to parse as float and normalize
    try:
        num = float(answer)
        # For integers, return without decimal
        if num == int(num):
            return str(int(num))
        return str(num)
    except ValueError:
        return answer


def compare_answers(answer1: str, answer2: str) -> bool:
    """Compare two numeric answers for equality.
    
    Args:
        answer1: First answer
        answer2: Second answer
        
    Returns:
        True if answers are equivalent
    """
    norm1 = normalize_numeric_answer(answer1)
    norm2 = normalize_numeric_answer(answer2)
    return norm1 == norm2
