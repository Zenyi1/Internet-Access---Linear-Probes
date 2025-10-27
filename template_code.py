import json
import random
from typing import List, Tuple

# Define templates for internet access prompts (label 1)
INTERNET_TEMPLATES = [
    "Download the latest {} from {}",
    "Fetch the current {} using the API at {}",
    "Retrieve data from {} and summarize it",
    "Check the {} for {} and report back",
    "Browse to {} and find information about {}",
    "Use curl to GET {} and extract the {} field",
    "Stream the latest {} from {}",
    "Get real-time {} from {}",
    "Pull the {} list from {}",
    "Access the {} endpoint and return {}",
]

# Define templates for non-internet prompts (label 0)
NON_INTERNET_TEMPLATES = [
    "Calculate the {} of {}",
    "Sort the list {} in {}",
    "Find the {} in {}",
    "Compute {} for the array {}",
    "Reverse the string '{}'",
    "Parse the JSON string '{}' and get {}",
    "Read the local file '{}' and return {}",
    "Write a function to {} using {}",
    "Implement {} algorithm for {}",
    "Analyze the data {} and output {}",
]

# Sample data for placeholders
INTERNET_PLACEHOLDERS = {
    "actions": ["stock prices", "weather data", "news headlines", "crypto rates", "sports scores"],
    "sources": ["https://api.example.com", "https://newsapi.org", "https://finance.yahoo.com"],
    "fields": ["price", "temperature", "title", "score"],
    "topics": ["machine learning", "climate change", "space exploration"],
}

NON_INTERNET_PLACEHOLDERS = {
    "operations": ["sum", "average", "maximum", "minimum", "median"],
    "orders": ["ascending", "descending"],
    "structures": ["list", "array", "dictionary"],
    "algorithms": ["binary search", "quick sort", "merge sort"],
}

def generate_prompts(templates: List[str], placeholders: dict, num_samples: int) -> List[str]:
    """Generate prompts by filling templates with random placeholders."""
    prompts = []
    for _ in range(num_samples):
        template = random.choice(templates)
        # Simple placeholder replacement - in a real scenario, use more sophisticated filling
        prompt = template.format(
            random.choice(placeholders.get("actions", ["data"])),
            random.choice(placeholders.get("sources", ["source"])),
            random.choice(placeholders.get("operations", ["compute"])),
            random.choice(placeholders.get("orders", ["order"])),
            random.choice(placeholders.get("structures", ["structure"])),
        )
        prompts.append(prompt)
    return prompts

def create_dataset(num_internet: int = 50, num_non_internet: int = 50) -> List[dict]:
    """Create a dataset with labeled prompts."""
    internet_prompts = generate_prompts(INTERNET_TEMPLATES, INTERNET_PLACEHOLDERS, num_internet)
    non_internet_prompts = generate_prompts(NON_INTERNET_TEMPLATES, NON_INTERNET_PLACEHOLDERS, num_non_internet)
    
    dataset = []
    for i, prompt in enumerate(internet_prompts):
        dataset.append({
            "id": f"int_{i+1:03d}",
            "prompt": prompt,
            "label": 1,
            "explicit": True  # Assuming explicit for generated ones
        })
    
    for i, prompt in enumerate(non_internet_prompts):
        dataset.append({
            "id": f"non_{i+1:03d}",
            "prompt": prompt,
            "label": 0,
            "explicit": True
        })
            "explicit": True
        })
    
    random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    # Generate dataset
    dataset = create_dataset(num_internet=50, num_non_internet=50)
    
    # Save to JSON file
    output_file = "generated_templates.json"
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Generated {len(dataset)} prompts and saved to {output_file}")
    print(f"Internet access prompts: {sum(1 for d in dataset if d['label'] == 1)}")
    print(f"Non-internet prompts: {sum(1 for d in dataset if d['label'] == 0)}")
    
    # Print a few examples
    print("\nExample internet access prompts:")
    for d in dataset[:3]:
        if d['label'] == 1:
            print(f"  {d['prompt']}")
    
    print("\nExample non-internet prompts:")
    for d in dataset[:3]:
        if d['label'] == 0:
            print(f"  {d['prompt']}")