"""Colorful logging utilities for meta-learning prompts and responses."""

import random


class PromptLogger:
    """Utility class for logging formatted prompts with colors."""

    # ANSI color codes
    COLORS = {
        'reset': '\033[0m',
        'bold': '\033[1m',
        'dim': '\033[2m',

        # Foreground colors
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',

        # Bright foreground colors
        'bright_black': '\033[90m',
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',

        # Background colors
        'bg_black': '\033[40m',
        'bg_red': '\033[41m',
        'bg_green': '\033[42m',
        'bg_yellow': '\033[43m',
        'bg_blue': '\033[44m',
        'bg_magenta': '\033[45m',
        'bg_cyan': '\033[46m',
        'bg_white': '\033[47m',
    }

    @staticmethod
    def _colorize(text: str, color: str, bold: bool = False) -> str:
        """Apply color to text."""
        if color not in PromptLogger.COLORS:
            return text
        prefix = PromptLogger.COLORS['bold'] if bold else ''
        return f"{prefix}{PromptLogger.COLORS[color]}{text}{PromptLogger.COLORS['reset']}"

    @staticmethod
    def _print_separator(char: str = "=", length: int = 100, color: str = "cyan", bold: bool = True):
        """Print a colored separator line."""
        line = char * length
        print(PromptLogger._colorize(line, color, bold))

    @staticmethod
    def _print_section_header(title: str, color: str = "bright_cyan", bold: bool = True):
        """Print a section header with decorative borders."""
        PromptLogger._print_separator("=", 100, color, bold)
        centered_title = f"  {title}  "
        print(PromptLogger._colorize(centered_title, color, bold))
        PromptLogger._print_separator("=", 100, color, bold)

    @staticmethod
    def _print_subsection(label: str, content: str, label_color: str = "yellow", content_color: str = "white", max_lines: int = None):
        """Print a labeled subsection."""
        print(PromptLogger._colorize(f"\n{label}:", label_color, bold=True))
        print(PromptLogger._colorize("-" * 100, "bright_black"))

        if max_lines:
            lines = content.split('\n')
            if len(lines) > max_lines:
                displayed_content = '\n'.join(lines[:max_lines])
                print(PromptLogger._colorize(displayed_content, content_color))
                print(PromptLogger._colorize(f"... ({len(lines) - max_lines} more lines)", "dim"))
            else:
                print(PromptLogger._colorize(content, content_color))
        else:
            print(PromptLogger._colorize(content, content_color))

    @staticmethod
    def log_student_generation(
        rollout_id: int,
        inner_iter: int,
        problem_idx: int,
        original_prompt: str,
        teacher_hints: str,
        augmented_prompt: str,
        response: str,
        is_heldout: bool = False
    ):
        """Log student generation with augmented prompt and response."""
        stage_name = "HELDOUT STUDENT GENERATION" if is_heldout else "STUDENT GENERATION"
        header_color = "bright_magenta" if is_heldout else "bright_green"

        PromptLogger._print_section_header(
            f"[Global step {rollout_id}] {stage_name} - Iter {inner_iter + 1} - Problem {problem_idx}",
            color=header_color
        )

        PromptLogger._print_subsection("Original Problem", original_prompt, "bright_yellow", "white")

        if teacher_hints:
            PromptLogger._print_subsection("Teacher Hints Applied", teacher_hints, "bright_cyan", "cyan")
        else:
            print(PromptLogger._colorize("\n[No teacher hints yet]", "dim"))

        PromptLogger._print_subsection("Full Augmented Prompt", augmented_prompt, "yellow", "white", max_lines=30)
        PromptLogger._print_subsection("Student Response", response, "bright_green", "green")

        PromptLogger._print_separator("=", 100, header_color)
        print()  # Extra newline for spacing

    @staticmethod
    def log_grading(
        rollout_id: int,
        inner_iter: int,
        problem_idx: int,
        problem: str,
        solution: str,
        rubric: str,
        grader_explanation: str,
        grade: float,
        is_heldout: bool = False
    ):
        """Log grading process with grader prompt and feedback."""
        stage_name = "HELDOUT GRADING" if is_heldout else "GRADING"
        header_color = "bright_magenta" if is_heldout else "bright_blue"

        PromptLogger._print_section_header(
            f"[Global step {rollout_id}] {stage_name} - Iter {inner_iter + 1} - Problem {problem_idx}",
            color=header_color
        )

        PromptLogger._print_subsection("Problem", problem, "bright_yellow", "white")
        PromptLogger._print_subsection("Student Solution", solution, "bright_green", "green")
        PromptLogger._print_subsection("Rubric", rubric, "yellow", "white")

        # Format grade with color based on performance
        if grade >= 0.8:
            grade_color = "bright_green"
        elif grade >= 0.5:
            grade_color = "bright_yellow"
        else:
            grade_color = "bright_red"

        grade_text = f"{grade:.3f}"
        print(PromptLogger._colorize(f"\nGRADE: {grade_text}", grade_color, bold=True))

        PromptLogger._print_subsection("Grader Explanation", grader_explanation, "bright_blue", "blue")

        PromptLogger._print_separator("=", 100, header_color)
        print()  # Extra newline for spacing

    @staticmethod
    def log_teacher_hint_generation(
        rollout_id: int,
        inner_iter: int,
        ctx_idx: int,
        teacher_prompt: str,
        previous_hints: str,
        new_hints: str,
        problem_grades: list[tuple[int, float]]
    ):
        """Log teacher hint generation process."""
        PromptLogger._print_section_header(
            f"[Global step {rollout_id}] TEACHER HINT GENERATION - Iter {inner_iter + 1} - Context {ctx_idx}",
            color="bright_yellow"
        )

        # Show problem grades summary
        if problem_grades:
            grades_summary = ", ".join([f"P{idx}: {grade:.2f}" for idx, grade in problem_grades])
            print(PromptLogger._colorize(f"\nProblem Grades: {grades_summary}", "yellow"))

        if previous_hints:
            PromptLogger._print_subsection("Previous Teacher Hints", previous_hints, "dim", "bright_black")
        else:
            print(PromptLogger._colorize("\n[First iteration - no previous hints]", "dim"))

        PromptLogger._print_subsection("Full Teacher Prompt", teacher_prompt, "yellow", "white", max_lines=40)
        PromptLogger._print_subsection("New Teacher Hints Generated", new_hints, "bright_yellow", "yellow")

        PromptLogger._print_separator("=", 100, "bright_yellow")
        print()  # Extra newline for spacing

    @staticmethod
    def log_iteration_summary(rollout_id: int, inner_iter: int, avg_reward: float, num_problems: int):
        """Log summary stats for an iteration."""
        PromptLogger._print_separator("-", 100, "cyan", bold=False)
        summary = f"[Global step {rollout_id} | Iter {inner_iter + 1}] Iteration Complete - Avg Reward: {avg_reward:.4f} ({num_problems} problems)"

        if avg_reward >= 0.8:
            color = "bright_green"
        elif avg_reward >= 0.5:
            color = "bright_yellow"
        else:
            color = "bright_red"

        print(PromptLogger._colorize(summary, color, bold=True))
        PromptLogger._print_separator("-", 100, "cyan", bold=False)
        print()

    @staticmethod
    def log_heldout_summary(rollout_id: int, avg_reward: float, num_contexts: int):
        """Log heldout evaluation summary."""
        PromptLogger._print_separator("*", 100, "bright_magenta", bold=True)
        summary = f"[Global step {rollout_id}] HELDOUT EVALUATION COMPLETE - Avg Reward: {avg_reward:.4f} ({num_contexts} contexts)"

        if avg_reward >= 0.8:
            color = "bright_green"
        elif avg_reward >= 0.5:
            color = "bright_yellow"
        else:
            color = "bright_red"

        print(PromptLogger._colorize(summary, color, bold=True))
        PromptLogger._print_separator("*", 100, "bright_magenta", bold=True)
        print()
