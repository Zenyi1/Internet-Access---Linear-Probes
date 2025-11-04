import json
import random
import string

# Generate a random alias
def random_alias(length=3):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

# Tangled alias generation for one command
def tangle_alias_command(command, min_alias=4):
    words = command.split()
    num_aliases = min(min_alias, len(words))
    
    # Pick words for aliases
    selected_for_alias = random.sample(words, num_aliases)
    
    # Create aliases for selected words
    alias_map = {}
    for word in selected_for_alias:
        alias_map[word] = random_alias()
    
    # Tangle the aliases (shuffle references)
    tangled_order = selected_for_alias.copy()
    random.shuffle(tangled_order)
    
    # Build alias definitions
    alias_defs = []
    for i, word in enumerate(selected_for_alias):
        ref = tangled_order[i]
        alias_defs.append(f"alias {alias_map[word]}='{ref}'")
    
    alias_definitions = " ; ".join(alias_defs)
    
    # Replace words in command with aliases where applicable
    aliased_cmd_words = [alias_map.get(word, word) for word in words]
    aliased_cmd = " ".join(aliased_cmd_words)
    
    return f"{alias_definitions} ; {aliased_cmd}"

# Input/output files
input_file = "new_baseline.jsonl"
output_file = "commands_tangled_augmented_baseline2.jsonl"

# Load commands
with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

augmented_data = []

# Process in batches of 2
i = 0
while i < len(data):
    if i + 1 < len(data):
        cmd_pair = [data[i], data[i+1]]
        # Generate 5 variants for each command (including original)
        for idx, entry in enumerate(cmd_pair):
            for _ in range(5):
                new_entry = entry.copy()
                new_entry["command"] = tangle_alias_command(entry["command"])
                augmented_data.append(new_entry)
        i += 2
    else:
        # Odd last command, just generate 5 variants
        entry = data[i]
        for _ in range(5):
            new_entry = entry.copy()
            new_entry["command"] = tangle_alias_command(entry["command"])
            augmented_data.append(new_entry)
        i += 1

# Save to output JSON
with open(output_file, "w") as f:
    for entry in augmented_data:
        f.write(json.dumps(entry) + "\n")

print(f"Generated {len(augmented_data)} tangled alias commands in {output_file}")
