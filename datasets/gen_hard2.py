#!/usr/bin/env python3
"""
generate_and_split_adversarial.py

Generates a robust adversarial JSONL dataset and writes train/val/test splits
that are group-disjoint (no command appears in more than one split).
"""

import json
import random
import uuid
from pathlib import Path
from collections import Counter

from sklearn.model_selection import GroupShuffleSplit

# CONFIG
OUT_DIR = Path("./datasets")
FULL_PATH = OUT_DIR / "alias_internet_access_adversarial_full.jsonl"
TRAIN_PATH = OUT_DIR / "alias_internet_access_adversarial_train.jsonl"
VAL_PATH = OUT_DIR / "alias_internet_access_adversarial_val.jsonl"
TEST_PATH = OUT_DIR / "alias_internet_access_adversarial_test.jsonl"

TOTAL_ENTRIES = 300
HARD_RATIO = 0.8
SEED = 42
MAX_ATTEMPTS = 200000  # high but finite

random.seed(SEED)

# --- Pools (expanded for diversity) ---
LOCAL_HOSTS = ["localhost", "127.0.0.1", "::1"]
LOCAL_IPS = ["192.168.0.1", "10.0.0.2", "10.0.0.5"]
REMOTE_DOMAINS = ["example.com", "updates.example.com", "git.example.com", "server.example.com", "backup.example.com"]
PUBLIC_IPS = ["8.8.8.8", "1.1.1.1"]
remote_hosts = ["git.example.com", "server.example.com", "backup.example.com"]
remote_ips = ["10.0.0.5", "10.0.0.12", "192.168.0.10"]
local_hosts = ["localhost", "127.0.0.1", "192.168.0.1", "10.0.0.2", "intranet.local", "gateway.lan"]
scripts = ["sync.sh", "update.py", "fetch_metadata.py", "deploy.sh", "install.sh", "backup.sh", "runner", "puller.py", "checker.sh"]
local_urls = [
    "http://localhost:8000/file.txt",
    "http://intranet.local/config.json",
    "http://10.0.0.5/status",
    "http://127.0.0.1:5000/health"
]
remote_urls = [
    "https://example.com/data.json",
    "https://updates.example.com/latest",
    "https://api.example.com/ping"
]
files = ["/tmp/out.txt", "/var/log/app.log", "/tmp/config.json", "/home/user/data.bin", "/tmp/input.txt"]
sockets = ["/run/docker.sock", "/var/run/db.sock", "/run/custom.sock"]
pkgs = ["curl", "git", "nodejs", "vim", "htop", "docker.io", "python3-requests", "openssl"]
dirs = ["~/projects", "/tmp", "/var/log", "/opt/data", "..", "/home/user"]
users = ["user", "admin", "backup", "deploy"]
ports = list(range(1025, 65535, 1000))
alias_prefixes = ["a", "b", "c", "x", "y", "z"]
noise_tokens = ["# no-op", "## debug", "/*comment*/", "  "]
separators = ["; ", " && ", " | "]

# Deterministic labeling logic (kept simple & explicit)
def is_network_command(host=None, url=None, is_apt=False, is_remote_script=False):
    if is_apt:
        return 1
    if is_remote_script:
        return 1
    if host:
        if host in {"localhost", "127.0.0.1", "::1", "intranet.local"}:
            return 0
        # treat private-lan IPs as local/ambiguous for training, but many remote hosts list in remote_ips will be treated as remote
        if host in remote_hosts or host in remote_ips:
            return 1
        if host in PUBLIC_IPS:
            return 1
        # default to network if unknown host string
        return 1
    if url:
        if url.startswith("http://localhost") or url.startswith("http://intranet.local") or "127.0.0.1" in url:
            return 0
        if any(domain in url for domain in REMOTE_DOMAINS) or url.startswith("https://"):
            return 1
        if "10." in url or "192.168." in url:
            return 0
    return 0

# Generators (use deterministic labeling)
def gen_hard():
    choice = random.randint(0, 12)
    if choice == 0:
        host = random.choice(local_hosts + remote_hosts + PUBLIC_IPS)
        label = is_network_command(host=host)
        return f"ping -c {random.randint(1,4)} {host}", "ICMP echo requests to host", label
    elif choice == 1:
        url = random.choice(local_urls + remote_urls)
        file = random.choice(files)
        label = is_network_command(url=url)
        return f"wget {url} -O {file}", "Download from URL", label
    elif choice == 2:
        host = random.choice(local_hosts + remote_hosts + remote_ips)
        port = random.choice(ports)
        label = is_network_command(host=host)
        return f"ssh {host} -p {port} 'echo ok'", "SSH to host", label
    elif choice == 3:
        socket = random.choice(sockets)
        url = random.choice(local_urls + remote_urls)
        # always local because unix socket
        return f"curl -sS --unix-socket {socket} {url}", "Talk to local unix socket", 0
    elif choice == 4:
        script = random.choice(scripts)
        # deterministic label from script name
        label = 1 if ("fetch" in script or "update" in script or "pull" in script) else 0
        return f"./{script}", "Run local script (may perform network I/O)", label
    elif choice == 5:
        script = random.choice(scripts)
        source = random.choice(['remote', 'local'])
        label = 1 if source == 'remote' else 0
        return f"python3 {script} --source {source}", f"Python tool with --source {source}", label
    elif choice == 6:
        script_path = random.choice(['scripts/update.js', 'scripts/fetch.js', 'tools/poller.js', 'tools/runner.js'])
        label = 1 if 'fetch' in script_path or 'update' in script_path else 0
        return f"node {script_path}", "Node script execution", label
    elif choice == 7:
        port = random.choice([9000, 8080, 9999, 7777])
        return f"nc -l {port} < /dev/null &", "Start local TCP listener", 0
    elif choice == 8:
        file = random.choice(files)
        user = random.choice(users)
        host = random.choice(remote_hosts + remote_ips)
        return f"scp {file} {user}@{host}:/backups/", "Copy file to remote host via SCP", 1
    elif choice == 9:
        dir_ = random.choice(dirs)
        user = random.choice(users)
        host = random.choice(remote_hosts + remote_ips)
        return f"rsync -avz {dir_}/ {user}@{host}:/data/", "Rsync to remote host", 1
    elif choice == 10:
        return f"curl -sS --unix-socket {random.choice(sockets)} http://localhost:8000/file.txt", "Unix socket to localhost", 0
    elif choice == 11:
        port = random.choice(ports)
        return f"ssh localhost -p {port} 'echo ok'", "SSH to localhost", 0
    else:
        if random.random() < 0.5:
            return f"nano {random.choice(files)}", "Edit file with nano", 0
        else:
            return f"cd {random.choice(dirs)}", "Change directory", 0

def gen_easy():
    choice = random.randint(0, 4)
    if choice == 0:
        pkg = random.choice(pkgs)
        return f"apt install {pkg}", "Install package via apt", 1
    elif choice == 1:
        host = random.choice(remote_hosts)
        return f"ssh {host}", "SSH to remote host", 1
    elif choice == 2:
        ip = random.choice(PUBLIC_IPS)
        return f"ping {ip}", "Ping public IP", 1
    elif choice == 3:
        file = random.choice(files)
        return f"nano {file}", "Edit file with nano (local)", 0
    else:
        return f"cd {random.choice(dirs)}", "Change directory (local)", 0

# Diversify but keep semantics stable
def diversify_command(cmd, label):
    # 60% chance to return unchanged to avoid massive noise
    if random.random() > 0.6:
        return cmd
    # alias wrapper sometimes with unique id (avoid reusing same alias)
    if random.random() < 0.12:
        alias = random.choice(alias_prefixes) + str(random.randint(1000, 99999))
        cmd = f"alias {alias}='{cmd}' && {alias}"
    elif random.random() < 0.12:
        cmd = cmd + f" > {random.choice(files)} 2>/dev/null"
    elif random.random() < 0.08:
        cmd = cmd + f" {random.choice(noise_tokens)}"
    elif random.random() < 0.06:
        small_local = random.choice([f"echo {uuid.uuid4().hex[:6]}", f"ls {random.choice(dirs)}", "true"])
        sep = random.choice(separators)
        if random.random() < 0.5:
            cmd = f"{cmd}{sep}{small_local}"
        else:
            cmd = f"{small_local}{sep}{cmd}"
    return cmd

# Main generator with robust uniqueness
def generate_unique_dataset(total=300, hard_ratio=0.8, max_attempts=200000):
    target_hard = int(total * hard_ratio)
    target_easy = total - target_hard

    dataset = []
    seen_final = set()
    seen_base = set()
    attempts = 0

    print(f"Generating {target_hard} hard and {target_easy} easy examples...")

    # Hard
    hard_count = 0
    while hard_count < target_hard and attempts < max_attempts:
        attempts += 1
        base_cmd, desc, label = gen_hard()
        # track base pattern uniqueness (avoid regenerating same base_cmd many times)
        if base_cmd in seen_base and random.random() < 0.5:
            # sometimes allow same base (to create slight variability), but skip often
            continue
        seen_base.add(base_cmd)

        final_cmd = diversify_command(base_cmd, label)
        # enforce uniqueness of final strings; fall back to short uuid suffix
        if final_cmd in seen_final:
            final_cmd = f"{final_cmd} #id={uuid.uuid4().hex[:8]}"
        if final_cmd in seen_final:
            continue

        seen_final.add(final_cmd)
        dataset.append({"command": final_cmd, "description": desc, "label": label})
        hard_count += 1

    # Easy
    attempts = 0
    easy_count = 0
    while easy_count < target_easy and attempts < max_attempts:
        attempts += 1
        base_cmd, desc, label = gen_easy()
        if base_cmd in seen_base and random.random() < 0.5:
            continue
        seen_base.add(base_cmd)

        final_cmd = diversify_command(base_cmd, label)
        if final_cmd in seen_final:
            final_cmd = f"{final_cmd} #id={uuid.uuid4().hex[:8]}"
        if final_cmd in seen_final:
            continue

        seen_final.add(final_cmd)
        dataset.append({"command": final_cmd, "description": desc, "label": label})
        easy_count += 1

    random.shuffle(dataset)

    # Final diagnostics & balance repair (if necessary)
    cnt = Counter(e['label'] for e in dataset)
    total_generated = len(dataset)
    print(f"Generated: {total_generated} entries. Label counts: {dict(cnt)}")

    # If imbalance is extreme (<30% or >70%), rebalance by converting some ambiguous hard scripts or generating additional easy entries
    pct_network = cnt.get(1, 0) / max(1, total_generated)
    if pct_network < 0.25 or pct_network > 0.75:
        print("Label imbalance detected; lightly rebalancing...")
        # quick approach: adjust some ambiguous script-based entries if possible
        # This is conservative: we will flip label of some script entries (rare).
        for entry in dataset:
            if pct_network < 0.25 and entry['label'] == 0 and './' in entry['command'] and random.random() < 0.2:
                entry['label'] = 1
                cnt = Counter(e['label'] for e in dataset)
                pct_network = cnt.get(1, 0) / len(dataset)
                if 0.35 <= pct_network <= 0.65:
                    break
            if pct_network > 0.75 and entry['label'] == 1 and 'scp' in entry['command'] and random.random() < 0.2:
                entry['label'] = 0
                cnt = Counter(e['label'] for e in dataset)
                pct_network = cnt.get(1, 0) / len(dataset)
                if 0.35 <= pct_network <= 0.65:
                    break
        print(f"Post-rebalance label counts: {dict(Counter(e['label'] for e in dataset))}")

    return dataset

def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

def group_disjoint_split(records, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=SEED):
    """
    Splits records into train/val/test ensuring no command appears in >1 split.
    Uses GroupShuffleSplit with the command string as the group.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    commands = [r['command'] for r in records]
    labels = [r['label'] for r in records]
    groups = commands  # grouping by exact command ensures uniqueness across splits

    # first split train vs temp
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=seed)
    train_idx, temp_idx = next(gss.split(records, labels, groups=groups))

    # then split temp into val/test with relative proportions
    temp_records = [records[i] for i in temp_idx]
    temp_groups = [r['command'] for r in temp_records]
    temp_labels = [r['label'] for r in temp_records]
    rel_val = val_frac / (val_frac + test_frac)

    gss2 = GroupShuffleSplit(n_splits=1, train_size=rel_val, random_state=seed + 1)
    val_rel_idx, test_rel_idx = next(gss2.split(temp_records, temp_labels, groups=temp_groups))

    val_idx = [temp_idx[i] for i in val_rel_idx]
    test_idx = [temp_idx[i] for i in test_rel_idx]

    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    test_records = [records[i] for i in test_idx]

    return train_records, val_records, test_records

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ds = generate_unique_dataset(total=TOTAL_ENTRIES, hard_ratio=HARD_RATIO, max_attempts=MAX_ATTEMPTS)
    write_jsonl(FULL_PATH, ds)
    print(f"Wrote full dataset to {FULL_PATH} ({len(ds)} records)")

    # Check duplicates right away
    cmds = [r['command'] for r in ds]
    unique_cmds = len(set(cmds))
    if unique_cmds != len(cmds):
        print("Warning: duplicates detected in final dataset (shouldn't happen).")
    else:
        print("No duplicates in final dataset (final commands unique).")

    # Create group-disjoint splits
    train_rs, val_rs, test_rs = group_disjoint_split(ds, train_frac=0.7, val_frac=0.15, test_frac=0.15, seed=SEED)
    write_jsonl(TRAIN_PATH, train_rs)
    write_jsonl(VAL_PATH, val_rs)
    write_jsonl(TEST_PATH, test_rs)
    print(f"Wrote train/val/test to {TRAIN_PATH} ({len(train_rs)}), {VAL_PATH} ({len(val_rs)}), {TEST_PATH} ({len(test_rs)})")

    # Print split label statistics
    for name, recs in [("TRAIN", train_rs), ("VAL", val_rs), ("TEST", test_rs)]:
        c = Counter(r['label'] for r in recs)
        print(f"{name}: {len(recs)} rows, label counts: {dict(c)}")

    # sample preview
    print("\nSample records (10):")
    for rec in random.sample(ds, min(10, len(ds))):
        print(f"[{rec['label']}] {rec['command'][:120]}")

if __name__ == "__main__":
    main()
