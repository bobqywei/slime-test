# Meta-Learning Module

This module implements a meta-learning system where a **teacher model** learns to generate helpful additional information for problems through an inner optimization loop.

## Overview

The meta-learning training loop consists of:

1. **Outer Loop**: Train the teacher model via GRPO based on holdout performance
2. **Inner Loop**: For each batch:
   - Sample problems and split into train/holdout sets
   - Run multiple iterations where:
     - Generator creates solutions with current additional info
     - Grader evaluates solutions with rubrics + chain-of-thought
     - Teacher generates updated additional info based on feedback
   - Evaluate on holdout set with final additional info
   - Use holdout performance as reward for teacher training

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  OUTER LOOP (Main Training)                                 │
│  Trains: Teacher Model                                      │
│  Reward: Holdout problem performance                        │
└─────────────────────────────────────────────────────────────┘
    for rollout_id in range(num_rollout):
        │
        ┌───▼──────────────────────────────────────────────────┐
        │ 1. Sample batch & split into train/holdout          │
        │    - 75% train problems                              │
        │    - 25% holdout problems                            │
        └───┬──────────────────────────────────────────────────┘
            │
        ┌───▼──────────────────────────────────────────────────┐
        │ 2. INNER OPTIMIZATION LOOP                           │
        │    for inner_iter in range(num_inner_iterations):   │
        │        a. Generator: Sample solutions                │
        │        b. Grader: Score with rubric + CoT            │
        │        c. Teacher: Generate additional info          │
        └───┬──────────────────────────────────────────────────┘
            │
        ┌───▼──────────────────────────────────────────────────┐
        │ 3. HOLDOUT EVALUATION                                │
        │    - Use final additional_info from each train prob │
        │    - Generate & grade holdout solutions              │
        └───┬──────────────────────────────────────────────────┘
            │
        ┌───▼──────────────────────────────────────────────────┐
        │ 4. TEACHER TRAINING (GRPO)                          │
        │    - Reward = avg(holdout_grades)                   │
        │    - Train teacher on additional_info generation    │
        └────────────────────────────────────────────────────────┘
```

## Components

### 1. `ProblemsContext` (types.py)
Represents a single problem with its evolving state through the inner loop:
- `problem`: Original problem sample
- `rubric`: Problem-specific grading rubric
- `additional_info`: Current additional info from teacher
- `generator_rollouts`: Solutions generated so far
- `grader_feedbacks`: Grades + explanations received

### 2. `GraderEngine` (grader_engine.py)
Evaluates solutions using a grader model:
- Takes: problem, solution, rubric, ground truth
- Returns: grade (0-1) + chain-of-thought explanation
- Uses greedy decoding for consistent grading

### 3. `TeacherEngine` (teacher_engine.py)
Generates additional information to help solve problems:
- Takes: problem, rubric, past attempts, past feedback
- Returns: additional information string
- Trained via GRPO based on holdout performance

### 4. `MetaLearningRolloutManager` (rollout_manager.py)
Orchestrates the inner optimization loop:
- Manages problem contexts
- Coordinates generator, grader, teacher
- Produces training data for teacher model

## Usage

### Basic Example

```bash
python train.py \
    --use-meta-learning \
    --num-inner-iterations 3 \
    --n-samples-per-inner-prompt 4 \
    --inner-loop-train-ratio 0.75 \
    --rubric-key rubric \
    --additional-info-position prefix \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node 8 \
    --prompt-data train.parquet \
    --rollout-batch-size 32
```

### Required Arguments

- `--use-meta-learning`: Enable meta-learning mode
- `--rubric-key`: Key in dataset for problem rubrics (default: "rubric")
- `--prompt-data`: Training dataset with problems and rubrics

### Key Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-inner-iterations` | 3 | Inner loop iterations |
| `--n-samples-per-inner-prompt` | 4 | Generator samples per problem per iteration |
| `--inner-loop-train-ratio` | 0.75 | Fraction for training (rest is holdout) |
| `--grader-temperature` | 0.0 | Grader sampling temperature (greedy) |
| `--teacher-temperature` | 0.7 | Teacher sampling temperature |
| `--additional-info-position` | "prefix" | Where to inject additional info |

## Data Format

Your dataset should include:
- `messages` or `input_key`: Problem statement
- `label`: Ground truth answer
- `rubric`: Problem-specific grading rubric

Example:
```json
{
    "messages": [{"role": "user", "content": "Solve: 2x + 3 = 7"}],
    "label": {"answer": "x = 2"},
    "rubric": "Award 1.0 for correct answer. Award 0.5 for correct method but arithmetic error."
}
```

## Prompt Templates

### Grader Prompt
```
You are an expert grader evaluating a solution to a problem.

Problem: {problem}
Rubric: {rubric}
Ground Truth Answer: {ground_truth}
Student Solution: {solution}

Your task:
1. Evaluate the solution against the rubric
2. Provide a grade from 0.0 to 1.0
3. Explain your reasoning step-by-step

Format: GRADE: <score>
EXPLANATION: <reasoning>
```

### Teacher Prompt
```
You are an expert teacher helping students improve.

Problem: {problem}
Rubric: {rubric}

Previous Attempts and Feedback:
{attempts_and_grades}

Previous Additional Information: {previous_info}

Your task: Provide additional information, hints, or guidance.
```

## Customization

### Custom Grader Template
```python
from slime.meta_learning.prompt_templates import GraderPromptTemplate

custom_template = GraderPromptTemplate(template="""
Your custom grader prompt here.
Use {problem}, {solution}, {rubric}, {ground_truth}
""")
```

### Custom Teacher Template
```python
from slime.meta_learning.prompt_templates import TeacherPromptTemplate

custom_template = TeacherPromptTemplate(template="""
Your custom teacher prompt here.
""")
```

## Monitoring

The following metrics are logged to WandB:
- `meta_learning/inner_loop_time`: Time for inner loop
- `meta_learning/avg_inner_grade`: Average grade during inner loop
- `meta_learning/avg_holdout_grade`: Average holdout performance
- `meta_learning/num_train_problems`: Number of training problems
- `meta_learning/num_holdout_evals`: Number of holdout evaluations

## Advanced Usage

### Multi-Model Setup

The meta-learning system supports three models:

1. **Teacher Model** (being trained via RLHF)
   - Has both training servers (FSDP/Megatron) AND inference servers (SGLang)
   - Generates additional_info to help students solve problems
   - Configured via `--hf-checkpoint` and `--rollout-num-gpus`

2. **Student/Generator Model** (inference-only, NO training)
   - Only has inference servers (SGLang)
   - Generates problem solutions using teacher's additional_info
   - Can share teacher's inference pool OR have a separate pool

3. **Grader Model** (external, typically OpenAI API)
   - Evaluates student solutions with rubrics
   - Configured via `--grader-model` (defaults to GPT-4)

#### Shared Inference Pool (Default)

By default, teacher and student share the same inference pool:

```bash
python train.py \
    --use-meta-learning \
    --hf-checkpoint /models/teacher-7b \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 1
```

In this mode:
- **Teacher router**: `http://{sglang_router_ip}:{sglang_router_port}`
- **Student router**: Same as teacher (shared)

#### Separate Student Inference Pool

For dedicated student resources with a different model:

```bash
python train.py \
    --use-meta-learning \
    --hf-checkpoint /models/teacher-7b \
    --rollout-num-gpus 8 \
    --rollout-num-gpus-per-engine 1 \
    --student-checkpoint /models/student-7b \
    --student-num-gpus 4 \
    --student-num-gpus-per-engine 1
```

This creates:
- **Teacher**: 8 GPUs for training + inference (checkpoint: teacher-7b)
- **Student**: 4 GPUs for inference-only (checkpoint: student-7b)
- **Separate routers**: Auto-assigned ports for each

Router URLs:
- **Teacher**: `http://{sglang_router_ip}:{sglang_router_port}` (or `{teacher_port}`)
- **Student**: `http://{student_router_ip}:{student_router_port}`

#### External Grader Setup

```bash
export OPENAI_API_KEY=your_key_here

python train.py \
    --use-meta-learning \
    --grader-model gpt-4 \
    --grader-temperature 0.0 \
    --grader-concurrency 32 \
    ...
```

### Configuration Reference

| Argument | Default | Description |
|----------|---------|-------------|
| **Teacher Configuration** | | |
| `--hf-checkpoint` | (required) | Teacher model checkpoint path |
| `--rollout-num-gpus` | (required) | GPUs for teacher inference pool |
| `--teacher-port` | `sglang_router_port` | Teacher router port |
| **Student Configuration** | | |
| `--student-checkpoint` | Same as teacher | Student model checkpoint |
| `--student-num-gpus` | None (shared) | GPUs for separate student pool |
| `--student-num-gpus-per-engine` | Same as teacher | GPUs per student engine |
| `--student-router-port` | Auto-assigned | Student router port |
| `--student-router-ip` | Same as teacher | Student router IP |
| **Grader Configuration** | | |
| `--grader-model` | `gpt-4` | OpenAI model for grading |
| `--grader-temperature` | 0.0 | Grader temperature (greedy) |
| `--grader-concurrency` | 32 | Concurrent grader requests |

## Implementation Details

### Inner Loop Flow

For each training batch:
1. Sample N problems, split into 0.75*N train + 0.25*N holdout
2. For each train problem:
   - Initialize `additional_info = ""`
   - For k inner iterations:
     - Generate m solutions with current additional_info
     - Grade each solution → get grades + explanations
     - Teacher sees: problem, rubric, solutions, grades
     - Teacher generates: new additional_info
3. For each holdout problem × each train problem's final additional_info:
   - Generate 1 solution with that additional_info
   - Grade it → reward
4. Teacher's reward = average holdout grade across all holdout problems

### Teacher Training Data

Each training sample for the teacher:
- **Prompt**: Problem + rubric + solution history + grade history
- **Response**: The additional_info it generated
- **Reward**: Average grade on holdout problems using that additional_info
- **Groups**: Multiple additional_info variants per problem (for GRPO)

## Troubleshooting

### "Context not found for sample"
- Ensure rubric_key matches your dataset
- Check that problems have unique IDs

### "No samples to grade"
- Verify generator is producing responses

### Poor holdout performance
- Increase num_inner_iterations
- Adjust teacher_temperature (higher = more diverse)
- Check grader prompt quality

## References

- SLIME framework: https://github.com/sglang/slime
- GRPO algorithm: Group Relative Policy Optimization
- Meta-learning: Learning to learn paradigm
