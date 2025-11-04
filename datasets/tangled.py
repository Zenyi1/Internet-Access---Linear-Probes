import json
import random
import string

# Generate a random alias
def random_alias(length=3):
    return ''.join(random.choices(string.ascii_lowercase, k=length))

ALWAYS_ALIAS = True

# Generate a tangled alias command with full BoW preservation

# Generate a tangled alias command with random aliases per token (maximal BoW confusion)
def tangled_random_alias_command(cmd_words):
    token_aliases = [random_alias() for _ in cmd_words]
    alias_defs = [f"alias {alias}='{word}'" for alias, word in zip(token_aliases, cmd_words)]
    aliased_cmd = " ".join(token_aliases)
    return " ; ".join(alias_defs) + " ; " + aliased_cmd



# Input/output files
input_file = "new_baseline.jsonl"
output_file = "commands_tangled_bow.json"

# Load commands
with open(input_file, "r") as f:
    data = [json.loads(line) for line in f]

augmented_data = []

# Process in pairs
i = 0
while i < len(data):
    if i + 1 < len(data):
        cmd1_words = data[i]["command"].split()
        cmd2_words = data[i+1]["command"].split()
        
        combined_tokens = cmd1_words + cmd2_words

        # Generate 5 variants per command; for each variant regenerate aliases
        for entry, is_first in zip([data[i], data[i+1]], [True, False]):
            for _ in range(5):
                # ensure aliases are unique per line (small namespace, guard against collisions)
                combined_aliases = []
                seen = set()
                for _ in combined_tokens:
                    a = random_alias()
                    while a in seen:
                        a = random_alias()
                    seen.add(a)
                    combined_aliases.append(a)

                alias_defs = [f"alias {alias}='{word}'" for alias, word in zip(combined_aliases, combined_tokens)]
                alias_definitions = " ; ".join(alias_defs)

                aliases1 = combined_aliases[:len(cmd1_words)]
                aliases2 = combined_aliases[len(cmd1_words):]

                aliases = aliases1 if is_first else aliases2
                aliased_cmd = " ".join(aliases)
                final_cmd = f"{alias_definitions} ; {aliased_cmd}"
                new_entry = entry.copy()
                new_entry["command"] = final_cmd
                augmented_data.append(new_entry)
        
        i += 2

# Save to JSON
with open(output_file, "w") as f:
    for entry in augmented_data:
        f.write(json.dumps(entry) + "\n")

print(f"Generated {len(augmented_data)} tangled BoW-preserved commands in {output_file}")
