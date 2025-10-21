"""Utility functions for meta-learning."""

import copy
import random
from typing import Any

from slime.utils.types import ProblemsContext, Sample


def split_train_holdout(samples: list[list[Sample]], train_ratio: float = 0.8, seed: int = None) -> tuple:
    """Split samples into train and holdout sets.

    Args:
        samples: List of sample groups (one group per prompt)
        train_ratio: Fraction of samples to use for training
        seed: Random seed for reproducibility

    Returns:
        (train_samples, holdout_samples) tuple
    """
    if seed is not None:
        random.seed(seed)

    indices = list(range(len(samples)))
    random.shuffle(indices)

    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    holdout_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    holdout_samples = [samples[i] for i in holdout_indices]

    return train_samples, holdout_samples


def augment_prompt(original_prompt: str, additional_info: str, position: str = "suffix") -> str:
    """Augment prompt with additional information.

    Args:
        original_prompt: Original problem prompt
        additional_info: Additional information from teacher
        position: Where to place additional info ("prefix", "suffix", "system")

    Returns:
        Augmented prompt string
    """
    if not additional_info:
        return original_prompt

    if position == "prefix":
        return f"{additional_info}\n\n{original_prompt}"
    elif position == "suffix":
        user_prompt = original_prompt.split("<|im_start|>user", 1)[1].split("<|im_end|>", 1)[0].strip()
        new_user_prompt = user_prompt + f"\n\n# Hints/Guidance:\n{additional_info}"
        return original_prompt.replace(user_prompt, new_user_prompt)
    elif position == "system":
        # For chat templates, this would go in system message
        # For now, just prefix
        return f"[Additional Information]\n{additional_info}\n\n{original_prompt}"
    else:
        raise ValueError(f"Unknown position: {position}")


def create_problem_contexts(
    sample_groups: list[list[Sample]], n_problems_per_prompt_group: int
) -> list[list[ProblemsContext]]:
    spi_per_prompt_group = len(sample_groups[0])
    contexts = []

    for i in range(0, len(sample_groups), n_problems_per_prompt_group):
        group = sample_groups[i: i + n_problems_per_prompt_group]
        group = [p[0] for p in group]
        ctxs = [ProblemsContext(problems=group) for _ in range(spi_per_prompt_group)]
        contexts.append(ctxs)

    return contexts
