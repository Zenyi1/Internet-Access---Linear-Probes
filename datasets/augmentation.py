import json
import random
import re

input_file = "internet_access_binary_dataset.jsonl"
output_file = "augmented_internet_access_dataset.jsonl"
augmentation_factor = 5  # Each command will be augmented this many times (in addition to original)


def add_flags(cmd: str, label: int) -> str:
    """Add common flags to commands"""
    if label == 0:  # Non-internet commands
        flag_options = [
            " -v",  # verbose
            " -f",  # force
            " -r",  # recursive
            " -i",  # interactive
            " -a",  # all
            " -l",  # long format
            " -h",  # human readable
        ]
    else:  # Internet commands
        flag_options = [
            " -v",  # verbose
            " -q",  # quiet
            " -s",  # silent
            " -k",  # insecure (curl/wget)
            " -L",  # follow redirects
            " -O",  # output to file
            " -4",  # IPv4
            " -6",  # IPv6
        ]
    
    # Randomly add 1-2 flags
    num_flags = random.randint(1, 2)
    flags = random.sample(flag_options, min(num_flags, len(flag_options)))
    return cmd + "".join(flags)

def add_pipes(cmd: str) -> str:
    """Add common pipe operations"""
    pipe_options = [
        " | grep 'pattern'",
        " | head -n 10",
        " | tail -n 5",
        " | wc -l",
        " | sort",
        " | uniq",
        " | less",
        " | tee output.txt",
        " | awk '{print $1}'",
        " | sed 's/old/new/g'",
    ]
    pipe = random.choice(pipe_options)
    return cmd + pipe

def add_redirects(cmd: str) -> str:
    """Add output redirection"""
    redirect_options = [
        " > output.txt",
        " >> log.txt",
        " 2>&1",
        " 2> error.log",
        " &> all.log",
        " > /dev/null",
        " 2>/dev/null",
    ]
    redirect = random.choice(redirect_options)
    return cmd + redirect

def add_background_or_chain(cmd: str) -> str:
    """Add background execution or command chaining"""
    options = [
        " &",  # background
        " && echo 'Done'",  # chain on success
        " || echo 'Failed'",  # chain on failure
        "; echo 'Completed'",  # sequential
    ]
    addition = random.choice(options)
    return cmd + addition

def add_variable_assignment(cmd: str) -> str:
    """Prepend with variable assignment"""
    var_options = [
        "VAR=$(", ")",
        "RESULT=$(", ")",
        "OUTPUT=$(", ")",
        "DATA=$(", ")",
    ]
    var_start, var_end = random.choice(var_options)
    return var_start + cmd + var_end

def add_timeout_or_prefix(cmd: str, label: int) -> str:
    """Add timeout, nice, or other prefix commands"""
    if label == 1:  # Internet commands
        prefix_options = [
            "timeout 30s ",
            "time ",
            "nice -n 10 ",
        ]
    else:
        prefix_options = [
            "time ",
            "nice -n 10 ",
            "sudo ",
        ]
    prefix = random.choice(prefix_options)
    return prefix + cmd

def change_paths(cmd: str) -> str:
    """Change file paths to different variations"""
    # Replace simple filenames with different paths
    path_replacements = [
        (r'\bfile\.txt\b', random.choice(['data.txt', 'output.txt', 'input.txt', 'test.txt'])),
        (r'\b/tmp/\w+', random.choice(['/tmp/test', '/tmp/output', '/var/tmp/data'])),
        (r'\b\./\w+', random.choice(['./script.sh', './data.csv', './config.json'])),
        (r'\b~/\w+', random.choice(['~/documents/file', '~/data/test', '~/tmp/output'])),
    ]
    
    for pattern, replacement in path_replacements:
        if re.search(pattern, cmd):
            cmd = re.sub(pattern, replacement, cmd, count=1)
            break
    return cmd

def change_urls(cmd: str) -> str:
    """Change URLs to different variations for internet commands"""
    url_replacements = [
        (r'https?://[\w\.\-]+', random.choice([
            'https://example.com',
            'http://api.example.org',
            'https://data.example.net',
            'http://test.example.com',
        ])),
    ]
    
    for pattern, replacement in url_replacements:
        if re.search(pattern, cmd):
            cmd = re.sub(pattern, replacement, cmd, count=1)
            break
    return cmd

def change_hostnames(cmd: str) -> str:
    """Change hostnames/IPs for network commands"""
    hostname_patterns = [
        (r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', random.choice([
            '192.168.1.100',
            '10.0.0.50',
            '172.16.0.10',
        ])),
        (r'\b(user@)?[\w\-]+\.[\w\-]+\.[\w]+\b', random.choice([
            'user@server.example.com',
            'admin@host.domain.org',
            'root@remote.example.net',
        ])),
    ]
    
    for pattern, replacement in hostname_patterns:
        if re.search(pattern, cmd):
            cmd = re.sub(pattern, replacement, cmd, count=1)
            break
    return cmd


def augment_command(cmd: str, label: int) -> str:
    """Apply a random augmentation strategy to a command"""
    
    # Different strategies with weights
    strategies = [
        (add_flags, 3),  # More likely
        (add_pipes, 2),
        (add_redirects, 2),
        (add_background_or_chain, 1),
        (change_paths, 2),
        #(lambda c, l: add_timeout_or_prefix(c, l), 1),
    ]
    
    # Add label-specific strategies
    if label == 1:  # Internet commands
        strategies.extend([
            (change_urls, 2),
            (change_hostnames, 2),
        ])
    
    # Select strategy based on weights
    strategy_funcs, weights = zip(*strategies)
    selected_strategy = random.choices(strategy_funcs, weights=weights, k=1)[0]
    
    # Apply strategy
    if selected_strategy in [add_flags, add_timeout_or_prefix]:
        return selected_strategy(cmd, label)
    else:
        return selected_strategy(cmd)

# === MAIN ===

dataset = []

# Load original dataset
print(f"Loading dataset from {input_file}...")
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Skipping malformed line: {e}")

original_count = len(dataset)
print(f"Loaded {original_count} commands")

# Count original distribution
pos_count = sum(1 for item in dataset if item['label'] == 1)
neg_count = original_count - pos_count
print(f"  Positive (internet): {pos_count}")
print(f"  Negative (non-internet): {neg_count}")

# Augment dataset
augmented_dataset = []

# Add all original commands first
augmented_dataset.extend(dataset)

# Create augmented versions
print(f"\nGenerating {augmentation_factor}x augmented versions...")
for item in dataset:
    cmd = item['command']
    desc = item['description']
    label = item['label']
    
    for _ in range(augmentation_factor):
        augmented_cmd = augment_command(cmd, label)
        augmented_dataset.append({
            'command': augmented_cmd,
            'description': desc,
            'label': label
        })

# Shuffle the augmented dataset
random.shuffle(augmented_dataset)

# Save augmented dataset
with open(output_file, "w", encoding="utf-8") as f:
    for item in augmented_dataset:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

final_count = len(augmented_dataset)
final_pos = sum(1 for item in augmented_dataset if item['label'] == 1)
final_neg = final_count - final_pos

print(f"   Original size: {original_count}")
print(f"   Augmented size: {final_count} ({augmentation_factor + 1}x)")
print(f"   Positive (internet): {final_pos}")
print(f"   Negative (non-internet): {final_neg}")
print(f"output: {output_file}")
print(f"Run alias script")