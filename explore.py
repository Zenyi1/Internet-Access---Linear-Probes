import json
import pandas as pd
import re

# Load dataset
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

print(f"Networking commands from dataset 1: {len(df1[df1['category'] == 'Networking'])}")

# Second dataset
df2 = pd.read_json("hf://datasets/aelhalili/bash-commands-dataset/dataset.json")
df2 = df2.rename(columns={"prompt": "description", "response": "command"})

# === Define patterns for commands that ACTUALLY access the internet ===
# Excluded local-only network commands: ifconfig, ip, route, arp, netstat, ss, etc.
internet_access_patterns = [
    r"\b(curl|wget|ping|scp|ftp|sftp|ssh|rsync|telnet|nc|netcat|dig|host|nslookup|traceroute|nmap|whois|ncftp|lynx|links|elinks|sshfs|tftp)\b",
    r"http[s]?://",  # URLs
    r"\bgit\s+(clone|fetch|pull|push)\b",  # Git commands involving network
    r"\bapt(-get)?\s+(update|install|upgrade)\b",  # Package managers
    r"\b(yum|dnf|zypper|pacman)\s+(update|install|upgrade)\b",
    r"\bsnap\s+(install|refresh)\b",
    r"\bpip\s+install\b",
    r"\bnpm\s+(install|update)\b",
]

# Local-only network commands (to exclude)
local_network_only = [
    r"\b(ifconfig|ip\s+(addr|link|route)|route|arp|netstat|ss|hostname|hostnamectl|nmcli|nmtui|iwconfig|ethtool)\b",
]

def is_internet_access_command(command):
    """Check if command accesses the internet (not just local network config)"""
    if not isinstance(command, str):
        return False
    
    # Check if it's a local-only command
    for pattern in local_network_only:
        if re.search(pattern, command, re.IGNORECASE):
            return False
    
    # Check if it's an internet-accessing command
    for pattern in internet_access_patterns:
        if re.search(pattern, command, re.IGNORECASE):
            return True
    
    return False

# Filter dataset 2 for internet-accessing commands
df2["is_internet_command"] = df2["command"].apply(is_internet_access_command)
networking_df2 = df2[df2["is_internet_command"]].copy()
networking_df2["label"] = 1
networking_df2 = networking_df2[["command", "description", "label"]]

# Filter dataset 1 - manually check networking commands
networking_df1 = df1[df1['category'] == 'Networking'].copy()
networking_df1["is_internet_command"] = networking_df1["command"].apply(is_internet_access_command)
networking_df1 = networking_df1[networking_df1["is_internet_command"]].copy()
networking_df1["label"] = 1
networking_df1 = networking_df1[["command", "description", "label"]]

print(f"Internet-accessing commands from dataset 1: {len(networking_df1)}")

# Get non-networking commands as negative samples
non_networking_df1 = df1[df1["category"] != "Networking"].copy()
non_networking_df1["label"] = 0
non_networking_df1 = non_networking_df1[["command", "description", "label"]]

# Combine positive samples and remove duplicates
positive_df = pd.concat([networking_df1, networking_df2], ignore_index=True).drop_duplicates(subset=["command"])
pos_count = len(positive_df)

print(f"\nTotal internet-accessing commands: {pos_count}")

# Sample negative examples to match positive count
negative_df = non_networking_df1.sample(n=min(pos_count, len(non_networking_df1)), random_state=42)

# Combine and shuffle
final_df = pd.concat([positive_df, negative_df], ignore_index=True)
final_df = final_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to file
final_df.to_json("internet_access_binary_dataset.jsonl", orient="records", lines=True, force_ascii=False)

# Sanity check: show examples
print("\n🔹 Example positive samples (internet-accessing):")
print(positive_df.head(10)[["command", "description"]])

print("\n🔹 Example negative samples (non-internet):")
print(final_df[final_df['label'] == 0].head(5)[["command", "description"]])

print(f"\nFinal dataset size: {len(final_df)}")
print(f"Positive samples: {len(final_df[final_df['label'] == 1])}")
print(f"Negative samples: {len(final_df[final_df['label'] == 0])}")