import json
import pandas as pd

# Load dataset
dataset = []
with open("LINUX_TERMINAL_COMMANDS.jsonl", "r") as file:
    for line in file:
        dataset.append(json.loads(line))

# Convert to DataFrame
df = pd.DataFrame(dataset)

# Example: View category distribution
print(df.groupby("category").size())

# Example: Filter Networking commands
networking_cmds = df[df["category"] == "Networking"]
print(networking_cmds[["id", "command", "description"]])
