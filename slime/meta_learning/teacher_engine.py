"""Teacher engine for generating additional information."""

import asyncio
import logging
from argparse import Namespace
from typing import Optional

from slime.meta_learning.prompt_templates import TeacherPromptBuilder, TeacherPromptTemplate
from slime.utils.http_utils import post

logger = logging.getLogger(__name__)


class TeacherEngine:
    """Wrapper for teacher model inference."""

    def __init__(self, args: Namespace, template: Optional[TeacherPromptTemplate] = None):
        self.args = args
        self.template = template or TeacherPromptTemplate()
        self.prompt_builder = TeacherPromptBuilder(self.template)

        # Teacher server endpoint - uses teacher-specific router
        self.teacher_url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

        logger.info(f"TeacherEngine initialized with URL: {self.teacher_url}")

        # Teacher-specific sampling params
        self.sampling_params = {
            "temperature": getattr(args, "teacher_temperature", 0.7),
            "top_p": getattr(args, "teacher_top_p", 0.9),
            "top_k": getattr(args, "teacher_top_k", 50),
            "max_new_tokens": getattr(args, "teacher_max_tokens", 512),
        }

        self.semaphore = asyncio.Semaphore(getattr(args, "teacher_concurrency", 16))

    async def generate_additional_info(self, context: "ProblemsContext", inner_step_idx: int) -> str:  # noqa: F821
        """Generate additional information for a problem context."""
        # Build teacher prompt from context
        teacher_prompt = self.prompt_builder.build_from_context(context, inner_step_idx)

        # Call teacher model
        async with self.semaphore:
            try:
                payload = {"text": teacher_prompt, "sampling_params": self.sampling_params}
                response = await post(self.teacher_url, payload)

                teacher_response = response.get("text", "")

                # Parse response
                additional_info = self.template.parse_response(teacher_response)

                return additional_info

            except Exception as e:
                logger.error(f"Error generating additional info: {e}")
                # Return empty string on error
                return ""

    async def generate_additional_info_batch(self, contexts: list["ProblemsContext"], inner_step_idx: int) -> list[str]:  # noqa: F821
        """Generate additional information for multiple contexts in batch."""
        tasks = [self.generate_additional_info(ctx, inner_step_idx) for ctx in contexts]
        additional_infos = await asyncio.gather(*tasks)
        return additional_infos

    def close(self):
        """Cleanup resources."""
        pass
