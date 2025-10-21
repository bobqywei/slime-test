"""Meta-learning rollout manager with inner optimization loop."""

import asyncio
import copy
import logging
from typing import Any, Union

import ray
import torch

from slime.meta_learning.grader_engine import GraderEngine
from slime.meta_learning.prompt_logger import PromptLogger
from slime.meta_learning.teacher_engine import TeacherEngine
from slime.meta_learning.utils import (
    augment_prompt,
    create_problem_contexts,
    split_train_holdout,
)
from slime.ray.rollout import init_rollout_engines, _start_router
from slime.ray.rollout_data_source import RolloutDataSourceWithBuffer
from slime.ray.utils import Lock
from slime.rollout.sglang_rollout import generate
from slime.utils.http_utils import init_http_client
from slime.utils.metric_checker import MetricChecker
from slime.utils.misc import load_function
from slime.utils.ray_utils import Box
from slime.utils.types import ProblemsContext, Sample
from slime.utils.wandb_utils import init_wandb_secondary

logger = logging.getLogger(__name__)


@ray.remote
class MetaLearningRolloutManager:
    """Global step manager with inner optimization loop for meta-learning."""

    def __init__(self, args, pg, wandb_run_id):
        # Extract rollout pg if pg is a dict (for meta-learning with separate student pool)
        if isinstance(pg, dict):
            self.pgs = pg
            rollout_pg = pg["rollout"]
            student_pg = pg.get("student", None)
        else:
            self.pgs = {"rollout": pg}
            rollout_pg = pg
            student_pg = None

        # Initialize core attributes
        self.args = args
        self.pg = self.pgs  # Store the full pg dict to track both teacher and student pools

        # Start teacher/rollout router
        if args.sglang_router_ip is None or args.sglang_router_port is None:
            args.sglang_router_ip, args.sglang_router_port = _start_router(
                args,
                router_ip=args.sglang_router_ip,
                router_port=args.sglang_router_port,
                router_name="Teacher/Rollout"
            )

        # Initialize wandb and http client
        init_wandb_secondary(
            args, wandb_run_id, router_addr=f"http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        init_http_client(args)

        # Initialize data source
        self.data_source = RolloutDataSourceWithBuffer(args)

        # Load custom functions
        self.generate_rollout = load_function(self.args.rollout_function_path)
        self.eval_generate_rollout = load_function(self.args.eval_function_path)
        self.custom_reward_post_process_func = None
        if self.args.custom_reward_post_process_path is not None:
            self.custom_reward_post_process_func = load_function(self.args.custom_reward_post_process_path)
        print(f"import {self.args.rollout_function_path} as generate_rollout function.")
        print(f"import {self.args.eval_function_path} as eval_generate_rollout function.")

        if self.args.debug_train_only:
            raise NotImplementedError("debug_train_only is not supported in MetaLearningRolloutManager.")
        else:
            num_gpu_per_engine = min(args.rollout_num_gpus_per_engine, args.num_gpus_per_node)
            num_engines = args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            self.all_rollout_engines = [None] * num_engines
        self.num_new_engines = init_rollout_engines(args, rollout_pg, self.all_rollout_engines)
        self.nodes_per_engine = max(1, args.rollout_num_gpus_per_engine // args.num_gpus_per_node)
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        # Initialize metric checker
        self._metric_checker = MetricChecker.maybe_create(args)

        # Initialize student/generator inference pool if separate from teacher
        self._init_student_engines(args, self.pgs)

        # Initialize grader and teacher engines
        self.grader_engine = GraderEngine(args)
        self.teacher_engine = TeacherEngine(args)

        # Inner loop configuration
        self.num_inner_iterations = args.num_inner_iterations
        self.inner_n_problems_per_prompt = args.n_problems_per_prompt  # default 2
        self.train_ratio = (args.rollout_batch_size - args.heldout_ipb) / args.rollout_batch_size

        # Prompt logging configuration
        self.log_prompts = getattr(args, "log_meta_learning_prompts", True)
        self.log_prompts_sample_rate = getattr(args, "log_prompts_sample_rate", 1.0)  # Log all by default

        logger.info(
            f"MetaLearningRolloutManager initialized with {self.num_inner_iterations} inner iterations"
        )
        logger.info(
            f"Teacher router: http://{args.sglang_router_ip}:{args.sglang_router_port}"
        )
        logger.info(
            f"Student router: http://{args.student_router_ip}:{args.student_router_port}"
        )

    @property
    def rollout_engines(self):
        """Get rollout inference engines (accounting for multi-node serving)."""
        # when doing multi-node serving, we will only send request to node-0 for each engine.
        return self.all_rollout_engines[:: self.nodes_per_engine]

    def _init_student_engines(self, args, pg):
        """Initialize student/generator inference engines (separate from teacher)."""
        print(
            f"Meta-learning: Creating separate student inference pool with "
            f"{args.student_num_gpus} GPUs ({args.student_num_gpus_per_engine} per engine)"
        )

        # Start student router if needed
        if args.student_router_port is None:
            args.student_router_ip, args.student_router_port = _start_router(
                args,
                router_ip=args.student_router_ip,
                router_port=None,
                router_name="Student"
            )

        if isinstance(pg, dict) and "student" in pg:
            student_pg = pg["student"]
        else:
            raise ValueError("Student placement group not found in pg dict")

        student_args = copy.copy(args)
        student_args.hf_checkpoint = args.student_checkpoint
        assert student_args.hf_checkpoint is not None, "Student checkpoint must be specified for separate student pool"

        student_args.sglang_router_ip = args.student_router_ip
        student_args.sglang_router_port = args.student_router_port
        student_args.rollout_num_gpus = args.student_num_gpus
        student_args.rollout_num_gpus_per_engine = args.student_num_gpus_per_engine

        self.student_args = student_args

        # Initialize student engines
        num_gpu_per_engine = min(student_args.rollout_num_gpus_per_engine, student_args.num_gpus_per_node)
        num_engines = student_args.student_num_gpus // num_gpu_per_engine
        self.all_student_engines = [None] * num_engines
        self.num_new_student_engines = init_rollout_engines(
            student_args, student_pg, self.all_student_engines
        )
        self.nodes_per_student_engine = max(
            1, student_args.rollout_num_gpus_per_engine // student_args.num_gpus_per_node
        )
        self.student_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

    @property
    def student_engines(self):
        """Get student inference engines (accounting for multi-node serving)."""
        if hasattr(self, 'all_student_engines'):
            return self.all_student_engines[:: self.nodes_per_student_engine]
        return self.rollout_engines  # Fallback to shared pool

    async def generate_with_inner_loop(self, rollout_id):
        """Main entry point: runs inner loop + generates training data for teacher."""
        import time

        start_time = time.time()

        try:
            all_samples = self.data_source.get_samples(self.args.rollout_batch_size)
            train_samples, holdout_samples = split_train_holdout(
                all_samples, self.train_ratio, seed=rollout_id
            )

            logger.info(
                f"Global step {rollout_id}: Split into {len(train_samples)} train, {len(holdout_samples)} holdout"
            )

            problem_contexts = await self._run_inner_loop(train_samples, rollout_id)

            await self._evaluate_holdout(holdout_samples, problem_contexts, rollout_id=rollout_id)

            self._log_heldout_metrics(problem_contexts, rollout_id)

            teacher_train_data = self._convert_to_teacher_train_data(problem_contexts)

            # self._log_inner_loop_metrics(rollout_id, problem_contexts, holdout_results, time.time() - start_time)

            return Box(ray.put(teacher_train_data))

        except Exception as e:
            logger.error(f"Error in generate_with_inner_loop: {e}", exc_info=True)
            raise

    async def _run_inner_loop(self, train_samples: list[list[Sample]], rollout_id: int) -> list[ProblemsContext]:
        """Execute the inner optimization loop."""
        # Initialize problem contexts
        contexts = create_problem_contexts(train_samples, self.inner_n_problems_per_prompt)

        print(f"Global step {rollout_id}: Starting inner loop with {len(contexts)} prompt groups and {len(contexts[0])} spi per group")

        for inner_iter in range(self.num_inner_iterations):
            print(f"Global step {rollout_id}: Inner iteration {inner_iter + 1}/{self.num_inner_iterations}")

            await self._generate_with_additional_info(self.student_args, contexts, inner_iter, rollout_id=rollout_id)

            await self._grade_solutions(contexts, inner_iter, rollout_id=rollout_id)

            await self._update_additional_info(contexts, inner_iter, rollout_id=rollout_id)

            # Log metrics at the end of each iteration
            self._log_iteration_metrics(contexts, rollout_id, inner_iter)

        print(f"Global step {rollout_id}: Inner loop completed")
        return contexts

    async def _generate_with_additional_info(self, args, contexts: list[list[ProblemsContext]], inner_iter: int, heldout: bool = False, rollout_id: int = None):
        """Generate solutions using generator with augmented prompts."""
        import random
        from slime.rollout.sglang_rollout import GenerateState

        state = GenerateState(args)
        tasks = []

        # Store metadata for logging
        generation_metadata = []

        for prompt_group_idx, prompt_group_contexts in enumerate(contexts):
            for spi_idx, prompt_ctx in enumerate(prompt_group_contexts):
                problems = prompt_ctx.heldout_problems if heldout else prompt_ctx.problems
                for problem_idx, problem in enumerate(problems):
                    teacher_hints = (
                        prompt_ctx.additional_info_per_inner_step[-1]
                        if prompt_ctx.additional_info_per_inner_step else ""
                    )
                    augmented_prompt = augment_prompt(problem.prompt, teacher_hints)

                    sample = Sample(prompt=augmented_prompt, metadata={
                        "prompt_group_index": prompt_group_idx,
                        "spi_index": spi_idx,
                        "problem_index": problem_idx,
                        "original_prompt": problem.prompt,
                        "teacher_hints": teacher_hints,
                    })
                    tasks.append(generate(args, sample, state.sampling_params))
                    generation_metadata.append({
                        "original_prompt": problem.prompt,
                        "teacher_hints": teacher_hints,
                        "augmented_prompt": augmented_prompt,
                        "prompt_group_idx": prompt_group_idx,
                        "spi_idx": spi_idx,
                        "problem_idx": problem_idx,
                    })

        results = await asyncio.gather(*tasks)

        # Process results and log sample prompts
        for idx, result in enumerate(results):
            prompt_group_idx = result.metadata["prompt_group_index"]
            spi_idx = result.metadata["spi_index"]
            problem_idx = result.metadata["problem_index"]
            ctx = contexts[prompt_group_idx][spi_idx]

            if heldout:
                ctx.heldout_problems[problem_idx].response = result.response
                ctx.heldout_problems[problem_idx].grader_cot = result.grader_cot
                ctx.heldout_problems[problem_idx].reward = result.reward
                ctx.heldout_problems[problem_idx].loss_mask = result.loss_mask
            else:
                if problem_idx not in ctx.problem_rollouts_dict:
                    ctx.problem_rollouts_dict[problem_idx] = []
                ctx.problem_rollouts_dict[problem_idx].append(result)

            # Log a sample of prompts based on sampling rate
            if self.log_prompts and rollout_id is not None and random.random() < self.log_prompts_sample_rate:
                # Log first problem of first context as example
                if idx == 0:
                    meta = generation_metadata[idx]
                    PromptLogger.log_student_generation(
                        rollout_id=rollout_id,
                        inner_iter=inner_iter,
                        problem_idx=meta["problem_idx"],
                        original_prompt=meta["original_prompt"],
                        teacher_hints=meta["teacher_hints"],
                        augmented_prompt=meta["augmented_prompt"],
                        response=result.response,
                        is_heldout=heldout
                    )

        logger.debug(f"Generated {len(results)} solutions across {len(contexts)} problems")

    async def _grade_solutions(self, contexts: list[list[ProblemsContext]], inner_iter: int = 0, rollout_id: int = None):
        """Grade solutions using grader model."""
        import random

        # Collect all recent samples (from this iteration)
        samples_to_grade = []
        for ctx_group in contexts:
            for ctx in ctx_group:
                for problem_idx, problem in enumerate(ctx.problems):
                    if problem_idx in ctx.problem_rollouts_dict:
                        recent_rollout = ctx.problem_rollouts_dict[problem_idx][-1]
                        samples_to_grade.append((problem, recent_rollout))

        await self.grader_engine.grade_samples(samples_to_grade)

        # Log a sample of grading results
        if self.log_prompts and rollout_id is not None and samples_to_grade and random.random() < self.log_prompts_sample_rate:
            problem, sample = samples_to_grade[0]  # Log first one as example
            PromptLogger.log_grading(
                rollout_id=rollout_id,
                inner_iter=inner_iter,
                problem_idx=0,
                problem=problem.prompt,
                solution=sample.response,
                rubric=problem.rubric or "N/A",
                grader_explanation=sample.grader_cot or "N/A",
                grade=sample.reward if sample.reward is not None else 0.0,
                is_heldout=False
            )

        logger.debug(f"Graded {len(samples_to_grade)} solutions")

    async def _grade_heldout_solutions(self, contexts: list[list[ProblemsContext]], rollout_id: int = None):
        """Grade heldout solutions using grader model."""
        import random

        # Collect all heldout samples
        samples_to_grade = []
        for ctx_group in contexts:
            for ctx in ctx_group:
                for problem in ctx.heldout_problems:
                    samples_to_grade.append((problem, problem))

        await self.grader_engine.grade_samples(samples_to_grade)

        # Log a sample of heldout grading results
        if self.log_prompts and rollout_id is not None and samples_to_grade and random.random() < self.log_prompts_sample_rate:
            problem, sample = samples_to_grade[0]  # Log first one as example
            PromptLogger.log_grading(
                rollout_id=rollout_id,
                inner_iter=self.num_inner_iterations - 1,  # Use final iteration
                problem_idx=0,
                problem=problem.prompt,
                solution=sample.response or "N/A",
                rubric=problem.rubric or "N/A",
                grader_explanation=sample.grader_cot or "N/A",
                grade=sample.reward if sample.reward is not None else 0.0,
                is_heldout=True
            )

        logger.debug(f"Graded {len(samples_to_grade)} heldout solutions")

    async def _update_additional_info(self, contexts: list[list[ProblemsContext]], inner_step_idx: int, rollout_id: int = None):
        """Update additional_info using teacher model."""
        import random

        flat_contexts = [ctx for ctx_group in contexts for ctx in ctx_group]
        additional_infos = await self.teacher_engine.generate_additional_info_batch(
            flat_contexts, inner_step_idx
        )

        # Log a sample of teacher hint generation
        if self.log_prompts and rollout_id is not None and flat_contexts and random.random() < self.log_prompts_sample_rate:
            ctx = flat_contexts[0]  # Log first context as example

            # Get teacher prompt before appending new hints
            teacher_prompt = self.teacher_engine.prompt_builder.build_from_context(ctx, inner_step_idx)

            # Get previous hints
            previous_hints = (
                ctx.additional_info_per_inner_step[-1]
                if ctx.additional_info_per_inner_step else ""
            )

            # Get problem grades for summary
            problem_grades = []
            for problem_idx, problem in enumerate(ctx.problems):
                if problem_idx in ctx.problem_rollouts_dict:
                    recent_rollout = ctx.problem_rollouts_dict[problem_idx][-1]
                    if hasattr(recent_rollout, 'reward') and recent_rollout.reward is not None:
                        problem_grades.append((problem_idx, recent_rollout.reward))

            new_hints = additional_infos[0]

            PromptLogger.log_teacher_hint_generation(
                rollout_id=rollout_id,
                inner_iter=inner_step_idx,
                ctx_idx=0,
                teacher_prompt=teacher_prompt,
                previous_hints=previous_hints,
                new_hints=new_hints,
                problem_grades=problem_grades
            )

        for ctx, new_info in zip(flat_contexts, additional_infos):
            ctx.additional_info_per_inner_step.append(new_info)

        print(f"Updated additional_info for {len(contexts)} problem groups with SPI {len(contexts[0])}")

    def _log_iteration_metrics(self, contexts: list[list[ProblemsContext]], rollout_id: int, inner_iter: int):
        """Log metrics for the current inner loop iteration."""
        import numpy as np

        all_rewards = []
        all_response_lengths = []
        all_additional_info_lengths = []

        # Collect metrics from all contexts
        for ctx_group in contexts:
            for ctx in ctx_group:
                # Collect rewards from the latest rollouts for each problem
                for problem_idx, rollouts in ctx.problem_rollouts_dict.items():
                    if rollouts:
                        latest_rollout = rollouts[-1]  # Get the most recent rollout
                        if hasattr(latest_rollout, 'reward') and latest_rollout.reward is not None:
                            all_rewards.append(latest_rollout.reward)
                        if hasattr(latest_rollout, 'response_length') and latest_rollout.response_length is not None:
                            all_response_lengths.append(latest_rollout.response_length)
                        elif hasattr(latest_rollout, 'response') and latest_rollout.response is not None:
                            all_response_lengths.append(len(latest_rollout.response))

                # Collect additional_info length (most recent)
                if ctx.additional_info_per_inner_step:
                    latest_info = ctx.additional_info_per_inner_step[-1]
                    all_additional_info_lengths.append(len(latest_info) if latest_info else 0)

        # Calculate statistics
        avg_reward = np.mean(all_rewards) if all_rewards else 0.0
        avg_response_length = np.mean(all_response_lengths) if all_response_lengths else 0.0
        avg_additional_info_length = np.mean(all_additional_info_lengths) if all_additional_info_lengths else 0.0

        # Print to CLI
        print(f"Global step {rollout_id} | Iteration {inner_iter + 1}/{self.num_inner_iterations} Metrics:")
        print(f"  Average Reward: {avg_reward:.4f} (n={len(all_rewards)})")
        print(f"  Average Response Length: {avg_response_length:.2f} (n={len(all_response_lengths)})")
        print(f"  Average Additional Info Length: {avg_additional_info_length:.2f} (n={len(all_additional_info_lengths)})")

        # Log iteration summary with colors
        if self.log_prompts:
            PromptLogger.log_iteration_summary(rollout_id, inner_iter, avg_reward, len(all_rewards))

    def _log_heldout_metrics(self, contexts: list[list[ProblemsContext]], rollout_id: int):
        """Log metrics from heldout evaluation.

        Metrics are averaged over heldout problems within each context first,
        then averaged over all contexts (across SPI and num_context_groups).
        """
        import numpy as np

        # First compute average reward and response length per context
        per_context_avg_rewards = []
        per_context_avg_response_lengths = []

        for ctx_group in contexts:
            for ctx in ctx_group:
                if ctx.heldout_problems:
                    # Collect metrics from all heldout problems in this context
                    heldout_rewards = []
                    heldout_response_lengths = []

                    for heldout_problem in ctx.heldout_problems:
                        if hasattr(heldout_problem, 'reward') and heldout_problem.reward is not None:
                            heldout_rewards.append(heldout_problem.reward)
                        if hasattr(heldout_problem, 'response_length') and heldout_problem.response_length is not None:
                            heldout_response_lengths.append(heldout_problem.response_length)
                        elif hasattr(heldout_problem, 'response') and heldout_problem.response is not None:
                            heldout_response_lengths.append(len(heldout_problem.response))

                    # Average over heldout problems within this context
                    if heldout_rewards:
                        per_context_avg_rewards.append(np.mean(heldout_rewards))
                    if heldout_response_lengths:
                        per_context_avg_response_lengths.append(np.mean(heldout_response_lengths))

        # Now average over all contexts (across SPI and context groups)
        overall_avg_reward = np.mean(per_context_avg_rewards) if per_context_avg_rewards else 0.0
        overall_avg_response_length = np.mean(per_context_avg_response_lengths) if per_context_avg_response_lengths else 0.0

        # Print to CLI
        print(f"Global step {rollout_id} | Heldout Evaluation Metrics:")
        print(f"  Average Reward: {overall_avg_reward:.4f} (n_contexts={len(per_context_avg_rewards)})")
        print(f"  Average Response Length: {overall_avg_response_length:.2f} (n_contexts={len(per_context_avg_response_lengths)})")

        # Log heldout summary with colors
        if self.log_prompts:
            PromptLogger.log_heldout_summary(rollout_id, overall_avg_reward, len(per_context_avg_rewards))

    async def _evaluate_holdout(
        self, holdout_samples: list[list[Sample]], contexts: list[list[ProblemsContext]], rollout_id: int = None
    ) -> list[dict[str, Any]]:
        """Evaluate on holdout set using final additional_info from train problems."""
        for ctx_group in contexts:
            for ctx in ctx_group:
                # Don't care about SPI here, just need the problems
                ctx.heldout_problems = [s[0] for s in holdout_samples]

        # Use final inner iteration for logging heldout
        final_iter = self.num_inner_iterations - 1
        await self._generate_with_additional_info(self.student_args, contexts, final_iter, heldout=True, rollout_id=rollout_id)
        await self._grade_heldout_solutions(contexts, rollout_id=rollout_id)

    def _convert_to_teacher_train_data(
        self, problem_contexts: list[list[ProblemsContext]]
    ) -> dict:
        """Convert to GRPO training data for teacher model."""
        from slime.rollout.sglang_rollout import GenerateState

        state = GenerateState(self.args)
        teacher_samples = []

        # For each train context, create a teacher training sample
        for group_idx, ctx_group_with_spi in enumerate(problem_contexts):
            num_spi = len(ctx_group_with_spi)
            num_heldout_problems = len(ctx_group_with_spi[0].heldout_problems)
            reward_samples_over_problems_and_spi = [
                [
                    ctx.heldout_problems[heldout_idx]
                    for ctx in ctx_group_with_spi
                ] for heldout_idx in range(num_heldout_problems)
            ] # [num_heldout_problems, spi]
            adv_over_problems_and_spi: list[list[float]] = [
                self._post_process_rewards(samples_over_spi)[1]
                for samples_over_spi in reward_samples_over_problems_and_spi
            ]
            # average advantages over heldout problems
            adv_per_spi = [
                sum(
                    adv_over_problems_and_spi[heldout_idx][spi_idx]
                    for heldout_idx in range(num_heldout_problems)
                ) / num_heldout_problems
                for spi_idx in range(num_spi)
            ] # [spi]

            for spi_idx, ctx in enumerate(ctx_group_with_spi):
                for inner_step_idx in range(self.num_inner_iterations):
                    teacher_prompt = self.teacher_engine.prompt_builder.build_from_context(ctx, inner_step_idx)
                    teacher_response = ctx.additional_info_per_inner_step[inner_step_idx]

                    prompt_tokens = state.tokenizer.encode(teacher_prompt)
                    response_tokens = state.tokenizer.encode(teacher_response)

                    teacher_sample = Sample(
                        prompt=teacher_prompt,
                        response=teacher_response,
                        response_length=len(response_tokens),
                        tokens=prompt_tokens + response_tokens,
                        reward=adv_per_spi[spi_idx],
                        loss_mask=None,
                        status=Sample.Status.COMPLETED,
                    )
                    teacher_samples.append(teacher_sample)

        # Convert to training data format
        return self._convert_samples_to_train_data(teacher_samples)
    
    def _post_process_rewards(self, samples: Union[list[Sample], list[list[Sample]]]):
        """Post-process rewards for training (copied from RolloutManager)."""
        if self.custom_reward_post_process_func is not None:
            return self.custom_reward_post_process_func(self.args, samples)

        raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
        if (
            self.args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
            and self.args.rewards_normalization
        ):
            # group norm
            rewards = torch.tensor(raw_rewards, dtype=torch.float)
            if rewards.shape[-1] == self.args.n_samples_per_prompt * self.args.rollout_batch_size:
                rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
            else:
                # when samples count are not equal in each group
                rewards = rewards.view(-1, rewards.shape[-1])
            mean = rewards.mean(dim=-1, keepdim=True)
            rewards = rewards - mean

            if self.args.advantage_estimator in ["grpo", "gspo"] and self.args.grpo_std_normalization:
                std = rewards.std(dim=-1, keepdim=True)
                rewards = rewards / (std + 1e-6)

            return raw_rewards, rewards.flatten().tolist()

        return raw_rewards, raw_rewards

    def _convert_samples_to_train_data(self, samples: Union[list[Sample], list[list[Sample]]]):
        """Convert inference generated samples to training data (copied from RolloutManager)."""
        raw_rewards, rewards = self._post_process_rewards(samples)

        assert len(raw_rewards) == len(samples)
        assert len(rewards) == len(samples)

        train_data = {
            "tokens": [sample.tokens for sample in samples],
            "response_lengths": [sample.response_length for sample in samples],
            # some reward model, e.g. remote rm, may return multiple rewards,
            # we could use key to select the reward.
            "rewards": rewards,
            "raw_reward": raw_rewards,
            "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
            "sample_indices": [sample.index for sample in samples],
        }

        # loss mask
        # TODO: compress the loss mask
        loss_masks = []
        for sample in samples:
            # always instantiate loss_mask if not provided
            if sample.loss_mask is None:
                sample.loss_mask = [1] * sample.response_length
            assert (
                len(sample.loss_mask) == sample.response_length
            ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
            loss_masks.append(sample.loss_mask)
        train_data["loss_masks"] = loss_masks

        # overwriting the raw reward
        if samples[0].metadata and "raw_reward" in samples[0].metadata:
            train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

        # For rollout buffer
        if samples[0].metadata and "round_number" in samples[0].metadata:
            train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

        # Add rollout log probabilities for off-policy correction
        if samples[0].rollout_log_probs is not None:
            train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

        if samples[0].train_metadata is not None:
            train_data["metadata"] = [sample.train_metadata for sample in samples]

        return train_data

    def get_rollout_engines_and_lock(self):
        """Get rollout engines and lock for teacher/rollout pool."""
        return self.rollout_engines, self.rollout_engine_lock, self.num_new_engines

    def save(self, rollout_id):
        self.data_source.save(rollout_id)

    def load(self, rollout_id=None):
        self.data_source.load(rollout_id)


    def offload(self):
        """Offload both teacher/rollout and student engines (if separate)."""
        offload_tasks = [engine.release_memory_occupation.remote() for engine in self.rollout_engines]
        return offload_tasks

    def onload(self, tags=None):
        """Onload both teacher/rollout and student engines (if separate)."""
        onload_tasks = [engine.resume_memory_occupation.remote(tags=tags) for engine in self.rollout_engines]
        return onload_tasks

    def dispose(self):
        """Cleanup resources for both teacher/rollout and student pools."""
        if self._metric_checker is not None:
            self._metric_checker.dispose()
        self.grader_engine.close()
        self.teacher_engine.close()
