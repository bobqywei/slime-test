"""Prompt templates for grader and teacher models in meta-learning."""

from typing import Any


class GraderPromptTemplate:
    """Template for grader model prompts."""

    DEFAULT_TEMPLATE = """You are an expert grader evaluating a solution to a problem.

Problem:
{problem}

Rubric:
{rubric}

Ground Truth Answer:
{ground_truth}

Student Solution:
{solution}

# Your task:
1. Evaluate the solution against the rubric
2. Provide a grade from 0.0 to 1.0 (where 1.0 is perfect)
3. Explain your reasoning step-by-step

# Format your response as:
EXPLANATION: <your chain-of-thought reasoning>
GRADE: <score>

# Now provide your evaluation:"""

    def __init__(self, template: str = None):
        self.template = template or self.DEFAULT_TEMPLATE

    def format(self, problem: str, solution: str, rubric: str, ground_truth: Any = None) -> str:
        """Format the grader prompt."""
        return self.template.format(
            problem=problem,
            solution=solution,
            rubric=rubric,
            ground_truth=ground_truth if ground_truth is not None else "N/A",
        )

    @staticmethod
    def parse_response(response: str) -> dict[str, Any] | None:
        """Parse grader model response into grade and explanation."""
        grade = 0.0
        explanation = ""

        if not "GRADE:" in response or not "EXPLANATION:" in response:
            print("[GraderPromptTemplate] Warning: Response missing expected format: ", response)
            return None
        
        explanation = response.split("GRADE:")[0].split("EXPLANATION:")[1].strip()
        grade_str = response.split("GRADE:")[1].strip().split()[0]
        grade = float(grade_str)

        return {"grade": grade, "explanation": explanation}


class TeacherPromptTemplate:
    """Template for teacher model prompts."""

    DEFAULT_TEMPLATE = """<|im_start|>user
You are an expert teacher helping students improve their problem-solving skills.

# Problem Set
{problem_section}

# Previous hints from teacher:
{previous_additional_info}

# Your task: Provide additional information, hints, or guidance that will help students solve this problem better. Be concise and focus on the most impactful information. Only output the hints intended for the student without any extra text. Do not directly provide the solution or any information only relevant to any specific problem (keep the hints general). You are only given the information provided above, do not ask or assume anything beyond that.

Now provide the hint and guidance below that pertains generally across all problems.<|im_end|><|im_start|>assistant"""

    PER_PROBLEM_SECTION = """
Problem:
{problem}
Response:
{recent_response}
Rubric:
{rubric}
Feedback:
{feedback}
Grade: {grade}
""".strip()

    def __init__(self, template: str = None):
        self.template = template or self.DEFAULT_TEMPLATE

    def format(
        self,
        problem_section: str,
        previous_additional_info: str = "",
    ) -> str:
        """Format the teacher prompt."""
        return self.template.format(
            problem_section=problem_section,
            previous_additional_info=previous_additional_info,
        )

    @staticmethod
    def parse_response(response: str) -> str:
        """Parse teacher model response to extract additional info."""
        response = response.strip()
        return response


class TeacherPromptBuilder:
    """Builder for constructing teacher prompts from ProblemsContext."""

    def __init__(self, template: TeacherPromptTemplate = None):
        self.template = template or TeacherPromptTemplate()
        self.per_problem_section = self.template.PER_PROBLEM_SECTION

    def build_from_context(self, context: "ProblemsContext", inner_step_idx: int) -> str:  # noqa: F821
        """Build teacher prompt from ProblemsContext."""
        problem_section = ""
        for i, problem in enumerate(context.problems):
            recent_response = context.problem_rollouts_dict[i][inner_step_idx - 1]
            problem_section += self.per_problem_section.format(
                problem=problem.prompt,
                recent_response=recent_response.response,
                rubric=problem.rubric,
                feedback=recent_response.grader_cot,
                grade=recent_response.reward,
            ) + "\n\n"

        previous_additional_info = (
            "No hints from teacher yet..."
        ) if inner_step_idx == 0 else context.additional_info_per_inner_step[inner_step_idx - 1]
        return self.template.format(
            problem_section=problem_section,
            previous_additional_info=previous_additional_info,
        )
