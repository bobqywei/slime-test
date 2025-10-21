"""Grader engine for evaluating solutions with rubrics using OpenAI API."""

import asyncio
import logging
import os
from argparse import Namespace
from typing import Any, Optional

from openai import AsyncOpenAI

from slime.meta_learning.prompt_templates import GraderPromptTemplate
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


class GraderEngine:
    """Wrapper for grader model inference using OpenAI API."""

    def __init__(self, args: Namespace, template: Optional[GraderPromptTemplate] = None):
        self.args = args
        self.template = template or GraderPromptTemplate()

        api_key = getattr(args, "openai_api_key", None) or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key must be provided via args.openai_api_key or OPENAI_API_KEY env var")

        self.client = AsyncOpenAI(api_key=api_key)

        # Model configuration
        self.model = getattr(args, "grader_model", "gpt-5-mini-2025-08-07")

        # Concurrency control
        self.semaphore = asyncio.Semaphore(getattr(args, "grader_concurrency", 32))

        # Retry configuration
        self.max_retries = getattr(args, "grader_max_retries", 3)
        self.retry_base_delay = getattr(args, "grader_retry_base_delay", 1.0)
        self.retry_max_delay = getattr(args, "grader_retry_max_delay", 60.0)

    async def grade_solution(
        self, sample: Sample, problem: str, rubric: str, ground_truth: Any = None
    ) -> dict[str, Any]:
        """
        Grade a solution with retry mechanism.

        Returns:
            dict with 'success' (bool) and optional 'error' (str) keys
        """
        # Build grader prompt
        grader_prompt = self.template.format(
            problem=problem, solution=sample.response, rubric=rubric, ground_truth=ground_truth
        )

        # Call OpenAI API with concurrency control and retry logic
        async with self.semaphore:
            last_error = None

            for attempt in range(self.max_retries):
                try:
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are an expert grader evaluating solutions based on provided rubrics."},
                            {"role": "user", "content": grader_prompt}
                        ],
                    )

                    grader_response = response.choices[0].message.content
                    parsed = self.template.parse_response(grader_response)
                    if parsed is None:
                        print(response)
                        continue
                    sample.grader_cot = parsed["explanation"]
                    sample.reward = parsed["grade"]

                    return {"success": True}

                except Exception as e:
                    last_error = e

                    # If this is not the last attempt, wait before retrying with exponential backoff
                    if attempt < self.max_retries - 1:
                        delay = min(
                            self.retry_base_delay * (2 ** attempt),
                            self.retry_max_delay
                        )
                        await asyncio.sleep(delay)

            # All retries exhausted
            error_msg = f"{type(last_error).__name__}: {str(last_error)}"
            sample.grader_cot = f"Error during grading after {self.max_retries} attempts: {error_msg}"
            sample.reward = 0.0

            return {
                "success": False,
                "error": error_msg,
                "sample_index": sample.index,
                "attempts": self.max_retries
            }

    async def grade_samples(self, samples: list[tuple[Sample, Sample]]) -> None:
        """
        Grade all samples and log failures in a batched manner after completion.
        """
        tasks = []

        for problem, recent_rollout in samples:
            task = self.grade_solution(
                sample=recent_rollout,
                problem=problem.prompt,
                rubric=problem.rubric,
                ground_truth=problem.label,
            )
            tasks.append(task)

        # Gather all results
        results = await asyncio.gather(*tasks)

        # Collect failures for batched logging
        failures = [result for result in results if not result["success"]]

        # Log failures in a batched manner
        if failures:
            print("\n" + "=" * 80)
            print(f"GRADING FAILURES SUMMARY: {len(failures)} out of {len(samples)} samples failed")
            print("=" * 80)

            for i, failure in enumerate(failures, 1):
                print(f"\nFailure {i}/{len(failures)}:")
                print(f"  Sample Index: {failure['sample_index']}")
                print(f"  Error: {failure['error']}")
                print(f"  Attempts: {failure['attempts']}")

            print("\n" + "=" * 80 + "\n")

    def close(self):
        pass
