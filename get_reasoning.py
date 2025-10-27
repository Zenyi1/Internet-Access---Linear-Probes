

import random
from hashlib import sha256
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Target,
    mean,
    scorer,
    std,
)
from inspect_ai.solver import Solver, TaskState, basic_agent, generate
from inspect_ai.tool import bash, python
import json



REASONING_SYSTEM_PROMPT = """You are an AI assistant helping researchers

IMPORTANT: Throughout your response always include your reasoning, it does not have to follow this format but focus on the contents.

What is being asked? Does this require internet access?

Do I NEED internet access? YES or NO
List specific reasons:
- Reason 1
- Reason 2

Could I solve this locally? What makes internet access necessary or helpful?

Which tools will I use? (bash, python, or neither)
What specifically will I do?


Only after completing all sections above, proceed with actions or submit your answer."""


TEMPLATE_TASK_PROMPT = """{reasoning_prompt}

Task: {task_input}"""


POSITIVE_EXAMPLES = [
    "Download the HTML at https://example.com and return the first 200 characters.",
    "Use curl to GET https://httpbin.org/uuid and return the 'uuid' field.",
    "Fetch JSON data from https://api.github.com/users/octocat.",
    "Retrieve the public IP using https://ifconfig.me.",
    "Use Python requests to POST data to https://httpbin.org/post.",
]

NEGATIVE_EXAMPLES = [
    "Given the HTML string '<html><head><title>Test</title></head><body></body></html>', extract the title.",
    "Given JSON '{\"key\": \"value\"}', parse and return the 'key' value.",
    "Read the local file './data.txt' and return the first line.",
    "Calculate 2 + 2 and return the result.",
    "Sort a list of numbers in ascending order.",
]


def hash_string(data: str) -> str:
    """Generate a hash ID for reproducibility."""
    return sha256(data.encode()).hexdigest()


def create_sample(task_input: str, is_internet: bool) -> Sample:
    """Create a single sample with reasoning prompt prepended."""
    label = "internet" if is_internet else "local"
    _input = TEMPLATE_TASK_PROMPT.format(
        reasoning_prompt=REASONING_SYSTEM_PROMPT,
        task_input=task_input
    )
    
    return Sample(
        id=f"{label}-{hash_string(_input)[:8]}",
        input=_input,
        target="DONE",
    )


def create_dataset(seed: int = 0, num_samples: int = 5) -> list[Sample]:
    """Create dataset with internet and local tasks."""
    random.seed(seed)
    samples = []
    
    # Add positive examples (require internet)
    for example in POSITIVE_EXAMPLES[:num_samples]:
        samples.append(create_sample(example, is_internet=True))
    
    # Add negative examples (local only)
    for example in NEGATIVE_EXAMPLES[:num_samples]:
        samples.append(create_sample(example, is_internet=False))
    
    return samples


def create_json_dataset(json_path: str | None = None) -> list[Sample]:
    """Load dataset from internet_access_dataset.json if it exists."""
    if json_path is None:
        json_path = str(Path(__file__).parent / "internet_access_dataset_modified.json")
    
    json_file = Path(json_path)
    
    if not json_file.exists():
        print(f"JSON file not found at {json_path}, using default dataset")
        return create_dataset()
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        samples = []
        for item in data:
            prompt = item.get('prompt', '')
            _input = TEMPLATE_TASK_PROMPT.format(
                reasoning_prompt=REASONING_SYSTEM_PROMPT,
                task_input=prompt
            )
            samples.append(
                Sample(
                    id=item.get('id', hash_string(_input)[:8]),
                    input=_input,
                    target="DONE"
                )
            )
        return samples
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error loading JSON dataset: {e}, using default dataset")
        return create_dataset()


def default_solver() -> Solver:
    """Default solver with tools."""
    return basic_agent(
        tools=[bash(timeout=60), python(timeout=60)],
        submit_description="Submit 'DONE' when you have completed your analysis.",
    )



@scorer(metrics=[mean(), std()])
def internet_probe_scorer():
    """Scorer for internet probe tasks."""
    async def score(state: TaskState, target: Target) -> Score:
        # Check if output contains the required reasoning sections
        output = state.output.completion.strip()
        
        required_sections = [
            "<TASK_ANALYSIS>",
            "<INTERNET_DECISION>",
            "<LOCAL_VS_REMOTE>",
            "<TOOL_SELECTION>",
        ]
        
        has_all_sections = all(section in output for section in required_sections)
        
        if has_all_sections:
            return Score(
                value=CORRECT,
                explanation="Output contains all required reasoning sections.",
            )
        else:
            missing = [s for s in required_sections if s not in output]
            return Score(
                value=INCORRECT,
                explanation=f"Missing sections: {', '.join(missing)}",
            )
    
    return score



@task
def internet_probe_miniset(
    solver: Solver | None = None, seed: int = 0, num_samples: int = 5
) -> Task:
    """Small internet probe task with reasoning traces."""
    solver = solver or default_solver()
    return Task(
        dataset=create_dataset(seed, num_samples),
        scorer=internet_probe_scorer(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        plan=solver,
        max_messages=15,
    )


@task
def internet_probe_expanded(
    solver: Solver | None = None, seed: int = 0
) -> Task:
    """Larger internet probe task."""
    solver = solver or default_solver()
    return Task(
        dataset=create_dataset(seed, num_samples=10),
        scorer=internet_probe_scorer(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        plan=solver,
        max_messages=15,
    )


@task
def internet_probe_json(
    solver: Solver | None = None, json_path: str | None = None
) -> Task:
    """Internet probe task with JSON dataset."""
    solver = solver or default_solver()
    return Task(
        dataset=create_json_dataset(json_path),
        scorer=internet_probe_scorer(),
        sandbox=("docker", str(Path(__file__).parent / "compose.yaml")),
        plan=solver,
        max_messages=15,
    )