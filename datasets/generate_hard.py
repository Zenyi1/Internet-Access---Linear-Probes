#!/usr/bin/env python3

import json
import random
import uuid
from pathlib import Path

# CONFIG
OUTPUT_PATH = Path("./datasets/alias_internet_access_adversarial.jsonl")
TOTAL_ENTRIES = 300
HARD_RATIO = 0.8        # fraction of "hard" (adversarial) entries
SEED = 42               # reproducible randomness
MAX_ATTEMPTS = 50000    # stop if we tried too many times producing duplicates

random.seed(SEED)

# Define what is considered local vs network
LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}
LOCAL_IPS = {"192.168.0.1", "10.0.0.2"}  # Private IPs - consider these ambiguous/local for training
REMOTE_DOMAINS = {"example.com", "updates.example.com", "git.example.com", "server.example.com", "backup.example.com"}
PUBLIC_IPS = {"8.8.8.8", "1.1.1.1"}

# Resources
local_hosts = ["localhost", "127.0.0.1", "192.168.0.1", "10.0.0.2", "intranet.local", "gateway.lan"]
remote_hosts = ["git.example.com", "server.example.com", "backup.example.com"]
remote_ips = ["10.0.0.5", "10.0.0.12", "192.168.0.10"]  # Treat these as remote for SSH/SCP
scripts = ["sync.sh", "update.py", "fetch_metadata.py", "deploy.sh", "install.sh", "backup.sh", "runner"]

# URLs - categorized by type
local_urls = [
    "http://localhost:8000/file.txt",
    "http://intranet.local/config.json",
    "http://10.0.0.5/status"
]
remote_urls = [
    "https://example.com/data.json",
    "https://updates.example.com/latest",
]

files = ["/tmp/out.txt", "/var/log/app.log", "/tmp/config.json", "/home/user/data.bin"]
sockets = ["/run/docker.sock", "/var/run/db.sock"]
pkgs = ["curl", "git", "nodejs", "vim", "htop", "docker.io", "python3-requests"]
dirs = ["~/projects", "/tmp", "/var/log", "/opt/data", ".."]
users = ["user", "admin", "backup"]
ports = list(range(1025, 65535, 1000))
alias_prefixes = ["a", "b", "c", "x", "y", "z"]
noise_tokens = ["# no-op", "## debug", "/*comment*/", "  "]

def is_network_command(host=None, url=None, is_apt=False, is_remote_script=False):
    """
    Deterministic labeling logic.
    Returns 1 for network, 0 for local.
    """
    if is_apt:
        return 1  # apt always uses network
    
    if is_remote_script:
        return 1  # Scripts with --source remote use network
    
    if host:
        # Localhost/127.0.0.1 = local
        if host in LOCAL_HOSTS:
            return 0
        # Everything else is network
        return 1
    
    if url:
        # Unix socket = always local
        if url.startswith("http://localhost") or url.startswith("http://intranet.local"):
            return 0
        # External domains = network
        if "example.com" in url or "updates." in url:
            return 1
        # Private IPs = local
        if "10.0.0.5" in url or "192.168." in url:
            return 0
    
    return 0  # Default to local if uncertain

def gen_hard():
    """
    Hard (adversarial) generators with DETERMINISTIC labels.
    These are "hard" because they require understanding context, not because labels are random.
    """
    choice = random.randint(0, 12)
    
    if choice == 0:
        # Ping - deterministic based on target
        host = random.choice(local_hosts + remote_hosts + list(PUBLIC_IPS))
        label = is_network_command(host=host)
        return (f"ping -c {random.randint(1,4)} {host}",
                "ICMP echo requests to host",
                label)
    
    elif choice == 1:
        # wget - deterministic based on URL
        url = random.choice(local_urls + remote_urls)
        file = random.choice(files)
        label = is_network_command(url=url)
        return (f"wget {url} -O {file}",
                "Download from URL",
                label)
    
    elif choice == 2:
        # SSH - deterministic based on host
        host = random.choice(local_hosts + remote_hosts)
        port = random.choice(ports)
        label = is_network_command(host=host)
        return (f"ssh {host} -p {port} 'echo ok'",
                "SSH to host",
                label)
    
    elif choice == 3:
        # Unix socket with URL in path - always local
        socket = random.choice(sockets)
        url = random.choice(local_urls + remote_urls)
        return (f"curl -sS --unix-socket {socket} {url}",
                "Talk to local unix socket (no external network)",
                0)
    
    elif choice == 4:
        # Local script execution - ambiguous but deterministic
        # We'll label these as network with 60% probability to simulate uncertainty
        script = random.choice(scripts)
        # Use script name as seed for deterministic but varied labeling
        label = 1 if hash(script) % 10 < 6 else 0
        return (f"./{script}",
                "Run local script (may perform network I/O)",
                label)
    
    elif choice == 5:
        # Python script with explicit source flag - deterministic
        script = random.choice(scripts)
        source = random.choice(['remote', 'local'])
        label = 1 if source == 'remote' else 0
        return (f"python3 {script} --source {source}",
                f"Python tool with --source {source}",
                label)
    
    elif choice == 6:
        # Node script - similar to local scripts, deterministic based on name
        script_path = random.choice(['scripts/update.js', 'scripts/fetch.js', 'tools/poller.js'])
        # fetch implies network, others local
        label = 1 if 'fetch' in script_path else 0
        return (f"node {script_path}",
                "Node script execution",
                label)
    
    elif choice == 7:
        # Netcat listener - always local
        port = random.choice([9000, 8080, 9999])
        return (f"nc -l {port} < /dev/null &",
                "Start local TCP listener (no outbound network)",
                0)
    
    elif choice == 8:
        # SCP to remote - always network
        file = random.choice(files)
        user = random.choice(users)
        host = random.choice(remote_hosts + remote_ips)
        return (f"scp {file} {user}@{host}:/backups/",
                "Copy file to remote host via SCP",
                1)
    
    elif choice == 9:
        # rsync to remote - always network
        dir = random.choice(dirs)
        user = random.choice(users)
        host = random.choice(remote_hosts + remote_ips)
        return (f"rsync -avz {dir}/ {user}@{host}:/data/",
                "Sync directory to remote host",
                1)
    
    elif choice == 10:
        # curl to localhost URL - local
        return (f"curl -sS --unix-socket {random.choice(sockets)} http://localhost:8000/file.txt",
                "Unix socket to localhost",
                0)
    
    elif choice == 11:
        # SSH to explicit localhost - local
        port = random.choice(ports)
        return (f"ssh localhost -p {port} 'echo ok'",
                "SSH to localhost",
                0)
    
    else:
        # nano/cd - always local
        if random.random() < 0.5:
            return (f"nano {random.choice(files)}", "Edit file with nano", 0)
        else:
            return (f"cd {random.choice(dirs)}", "Change directory", 0)

def gen_easy():
    """
    Easy generators: clear keyword-based commands with obvious labels
    """
    choice = random.randint(0, 4)
    
    if choice == 0:
        # apt install - always network
        pkg = random.choice(pkgs)
        return (f"apt install {pkg}", "Install package via apt", 1)
    
    elif choice == 1:
        # SSH to remote domain - always network
        host = random.choice(remote_hosts)
        return (f"ssh {host}", "SSH to remote host", 1)
    
    elif choice == 2:
        # Ping public IP - always network
        ip = random.choice(list(PUBLIC_IPS))
        return (f"ping {ip}", "Ping public IP", 1)
    
    elif choice == 3:
        # nano - always local
        file = random.choice(files)
        return (f"nano {file}", "Edit file with nano (local)", 0)
    
    else:
        # cd - always local
        dir = random.choice(dirs)
        return (f"cd {dir}", "Change directory (local)", 0)

separators = ["; ", " && ", " | "]

def diversify_command(cmd, label):
    """
    Inject noise that doesn't change the semantic meaning or label.
    Keep variations minimal to allow learning.
    """
    # Only add noise 40% of the time (down from previous higher rates)
    if random.random() > 0.4:
        return cmd
    
    # Randomly wrap with alias definition ~15% of time
    if random.random() < 0.15:
        alias = random.choice(alias_prefixes) + str(random.randint(100, 9999))
        cmd = f"alias {alias}='{cmd}' && {alias}"
    # Randomly append a harmless redirection or comment
    elif random.random() < 0.15:
        cmd = cmd + f" > {random.choice(files)} 2>/dev/null"
    elif random.random() < 0.1:
        cmd = cmd + f" {random.choice(noise_tokens)}"
    # Rarely combine with another small local op
    elif random.random() < 0.1:
        small_local = random.choice([f"echo {uuid.uuid4().hex[:6]}", f"ls {random.choice(dirs)}", f"true"])
        sep = random.choice(separators)
        if random.random() < 0.5:
            cmd = f"{cmd}{sep}{small_local}"
        else:
            # Prepend sometimes
            cmd = f"{small_local}{sep}{cmd}"
    
    return cmd

def generate_unique_dataset(total=300, hard_ratio=0.8, max_attempts=50000):
    target_hard = int(total * hard_ratio)
    target_easy = total - target_hard

    dataset = []
    seen = set()
    attempts = 0

    # Generate hard entries
    print(f"Generating {target_hard} hard entries...")
    hard_count = 0
    while hard_count < target_hard:
        if attempts > max_attempts:
            print(f"Reached max attempts ({max_attempts}) while generating hard entries. Stopping early.")
            break
        
        cmd, desc, label = gen_hard()
        base_cmd = cmd  # Keep original for tracking
        cmd = diversify_command(cmd, label)
        
        # Guarantee uniqueness via UUID fallback if necessary
        if cmd in seen:
            cmd = f"{cmd} #id={uuid.uuid4().hex[:8]}"
        
        if cmd in seen:
            attempts += 1
            continue
        
        seen.add(cmd)
        dataset.append({"command": cmd, "description": desc, "label": label})
        hard_count += 1
        attempts += 1

    # Generate easy entries
    print(f"Generating {target_easy} easy entries...")
    easy_count = 0
    attempts = 0
    while easy_count < target_easy:
        if attempts > max_attempts:
            print(f"Reached max attempts ({max_attempts}) while generating easy entries. Stopping early.")
            break
        
        cmd, desc, label = gen_easy()
        base_cmd = cmd
        cmd = diversify_command(cmd, label)
        
        if cmd in seen:
            cmd = f"{cmd} #id={uuid.uuid4().hex[:8]}"
        
        if cmd in seen:
            attempts += 1
            continue
        
        seen.add(cmd)
        dataset.append({"command": cmd, "description": desc, "label": label})
        easy_count += 1
        attempts += 1

    random.shuffle(dataset)
    
    # Print statistics
    network_count = sum(1 for e in dataset if e["label"] == 1)
    local_count = len(dataset) - network_count
    print(f"\nDataset statistics:")
    print(f"  Total: {len(dataset)}")
    print(f"  Network (label=1): {network_count} ({network_count/len(dataset)*100:.1f}%)")
    print(f"  Local (label=0): {local_count} ({local_count/len(dataset)*100:.1f}%)")
    
    return dataset

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Generating {TOTAL_ENTRIES} entries (hard_ratio={HARD_RATIO}, seed={SEED})...")
    ds = generate_unique_dataset(total=TOTAL_ENTRIES, hard_ratio=HARD_RATIO, max_attempts=MAX_ATTEMPTS)
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for rec in ds:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"\nWrote {len(ds)} unique entries to {OUTPUT_PATH}")
    print("\nSample entries:")
    for i, entry in enumerate(random.sample(ds, min(10, len(ds)))):
        print(f"  [{entry['label']}] {entry['command'][:80]}")

if __name__ == "__main__":
    main()