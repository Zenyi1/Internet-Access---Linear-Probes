import json
import re
import random


input_file = "augmented_internet_access_dataset.jsonl"
output_file = "alias_internet_access_dataset.jsonl"
alias_map_file = "alias_mapping2.json"

# If True, non-internet commands will have a single token replaced with an alias (default: False)
obfuscate_non_internet = True

# Internet-accessing keywords (commands that actually access the internet)
# Excludes local-only network commands like ifconfig, ip addr, route, arp, netstat
internet_keywords = [
    # Data transfer and web access
    "curl", "wget", "lynx", "links", "elinks",
    
    # Remote access and file transfer
    "ssh", "scp", "sftp", "ftp", "rsync", "sshfs", "tftp",
    
    # Network testing and queries
    "ping", "dig", "host", "nslookup", "traceroute", "whois",
    
    # Network utilities
    "telnet", "nc", "netcat", "nmap",
    
    # Version control
    "git",
    
    # Package managers
    "apt", "apt-get", "yum", "dnf", "zypper", "pacman", "snap", "pip", "npm", "yarn"
]

def alias_name_gen():
    i = 1
    while True:
        yield f"a{i}"
        i += 1

_alias_gen = alias_name_gen()

# regex for token boundaries (word boundaries)
def token_regex(word):
    return re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)

def safe_single_quote(s: str) -> str:
    """Escape single quotes inside single-quoted shell string: ' -> '"'"'"""
    return s.replace("'", "'\"'\"'")

def find_internet_tokens(cmd: str):
    """Return list of (start, end, matched_token_originalcase, keyword_lower) for all internet token occurrences."""
    occurrences = []
    for kw in internet_keywords:
        pattern = token_regex(kw)
        for m in pattern.finditer(cmd):
            occurrences.append((m.start(), m.end(), m.group(0), kw.lower()))
    # sort by start index so replacing left-to-right is simpler
    occurrences.sort(key=lambda x: x[0])
    # Filter overlapping occurrences: keep the earliest non-overlapping
    filtered = []
    last_end = -1
    for occ in occurrences:
        if occ[0] >= last_end:
            filtered.append(occ)
            last_end = occ[1]
    return filtered

def aliasify_internet_tokens(cmd: str, alias_mapping: dict):
    if not isinstance(cmd, str) or not cmd.strip():
        return "", cmd

    occurrences = find_internet_tokens(cmd)
    if not occurrences:
        return "", cmd

    # Replace occurrences from right-to-left to preserve indices
    alias_prefix_parts = []
    
    # Build segments for replacement
    segs = []
    prev_end = len(cmd)
    for start, end, orig_token, kw_lower in reversed(occurrences):
        alias = next(_alias_gen)
        alias_prefix_parts.append((alias, orig_token))
        segs.append((start, end, alias))
        prev_end = start
    
    # Build the new command
    segs.reverse()
    out_parts = []
    cur = 0
    for start, end, alias in segs:
        out_parts.append(cmd[cur:start])
        out_parts.append(alias)
        cur = end
    out_parts.append(cmd[cur:])  # tail
    replaced_cmd = "".join(out_parts)

    # Build alias prefix and update mapping
    alias_prefix_parts.reverse()
    for alias, orig_token in alias_prefix_parts:
        alias_mapping[alias] = {"original_token": orig_token, "style": "token_alias"}

    # Construct prefix string in same order
    prefix_str = "".join([
        f"alias {a}='{safe_single_quote(alias_mapping[a]['original_token'])}'; " 
        for a, _ in alias_prefix_parts
    ])

    return prefix_str, replaced_cmd

def aliasify_one_token_non_internet(cmd: str, alias_mapping: dict):
    """
    For a non-internet command, optionally replace one random token with an alias,
    and prefix the alias definition. Returns prefix_str, replaced_cmd
    """
    if not isinstance(cmd, str) or not cmd.strip():
        return "", cmd
    parts = cmd.split()
    if not parts:
        return "", cmd
    idx = random.randrange(len(parts))
    orig = parts[idx]
    alias = next(_alias_gen)
    parts[idx] = alias
    replaced_cmd = " ".join(parts)
    alias_mapping[alias] = {"original_token": orig, "style": "token_alias_non_internet"}
    prefix_str = f"alias {alias}='{safe_single_quote(orig)}'; "
    return prefix_str, replaced_cmd

alias_mapping = {}
skipped_lines = []
out_count = 0
in_count = 0
aliased_count = 0
unchanged_count = 0

print(f"Processing {input_file}...")

with open(input_file, "r", encoding="utf-8") as inf, open(output_file, "w", encoding="utf-8") as outf:
    for line_no, raw in enumerate(inf, start=1):
        raw = raw.strip()
        if not raw:
            continue
        in_count += 1
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"Skipping malformed JSON line {line_no}: {e}")
            skipped_lines.append({"line_no": line_no, "error": str(e), "raw_head": raw[:200]})
            continue

        cmd = obj.get("command", "")
        desc = obj.get("description", "")
        label = int(obj.get("label", 0))

        if label == 1:
            # aliasify internet-accessing tokens only
            prefix, replaced_cmd = aliasify_internet_tokens(cmd, alias_mapping)
            # if there were no internet tokens found, we still leave cmd as-is (no prefix)
            new_cmd = prefix + replaced_cmd if prefix else replaced_cmd
            if prefix:
                aliased_count += 1
            else:
                unchanged_count += 1
        else:
            # non-internet: default leave unchanged unless obfuscate_non_internet True
            if obfuscate_non_internet:
                prefix, replaced_cmd = aliasify_one_token_non_internet(cmd, alias_mapping)
                new_cmd = prefix + replaced_cmd
                aliased_count += 1
            else:
                new_cmd = cmd
                unchanged_count += 1

        # write output JSONL (keep description and label)
        out_obj = {"command": new_cmd, "description": desc, "label": label}
        outf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
        out_count += 1

# save alias mapping
with open(alias_map_file, "w", encoding="utf-8") as m:
    json.dump(alias_mapping, m, ensure_ascii=False, indent=2)


print(f"   Dataset: {output_file}")
print(f"   Alias mapping: {alias_map_file}")

if skipped_lines:
    print("\n⚠️  Example skipped line(s):")
    for s in skipped_lines[:3]:
        print(f"   Line {s['line_no']}: {s['error']}")