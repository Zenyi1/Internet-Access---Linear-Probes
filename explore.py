import json
import pandas as pd
import re


# Load dataset
import json

dataset = []
with open("LINUX_TERMINAL_COMMANDS.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        # Optional cleanup step if you still get Invalid \escape errors:
        line = line.replace("\\", "\\\\")
        try:
            dataset.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"Error decoding line: {e}")

# Convert to DataFrame
df1 = pd.DataFrame(dataset)

#networking_df1 = df1[df1['category'] == 'networking']

print(f"Networking commands from dataset 1: {len(df1[df1["category"] == "Networking"])}")



##Second dataset
df2 = pd.read_json("hf://datasets/aelhalili/bash-commands-dataset/dataset.json")
df2 = df2.rename(columns={"prompt": "description", "response": "command"})

# === 3. Define patterns for commands that access the internet ===
network_patterns = [
    r"\b(curl|wget|ping|scp|ftp|sftp|ssh|rsync|telnet|nc|netcat|dig|host|nslookup|traceroute|nmap|ifconfig|ip|route|arp|whois|scp|ncftp|lynx|links|elinks|sshfs|tftp)\b",
    r"http[s]?://",  # URLs
    r"\bgit\s+(clone|fetch|pull|push)\b",  # Git commands involving network
]

def is_network_command(command):
    if not isinstance(command, str):
        return False
    for pattern in network_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    return False

df2["is_network_command"] = df2["command"].apply(is_network_command)
networking_df2 = df2[df2["is_network_command"]].copy()
networking_df2["label"] = 1
networking_df2 = networking_df2[["command", "description", "label"]]

networking_df1 = df1[df1['category'] == 'Networking'].copy()
networking_df1["label"] = 1
networking_df1 = networking_df1[["command", "description", "label"]]

non_networking_df1 = df1[df1["category"].str.lower() != "networking"].copy()
non_networking_df1["label"] = 0
non_networking_df1 = non_networking_df1[["command", "description", "label"]]

positive_df = pd.concat([networking_df1, networking_df2], ignore_index=True).drop_duplicates(subset=["command"])
pos_count = len(positive_df)

negative_df = non_networking_df1.sample(n=pos_count, random_state=42)
final_df = pd.concat([positive_df, negative_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True) #shuffle

final_df.to_json("nonalias_network_binary_dataset2.jsonl", orient="records", lines=True, force_ascii=False)

# Optional sanity check: show some examples
print("\n🔹 Example positive samples (networking):")
print(final_df[final_df['label'] == 1].head(5)[["command", "description"]])

print("\n🔹 Example negative samples (non-networking):")
print(final_df[final_df['label'] == 0].head(5)[["command", "description"]])

print(len(final_df))