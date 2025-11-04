
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BitsAndBytesConfig # need to implement save best probe


MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
MAX_LENGTH = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Using device: {DEVICE}")

print(f"\nLoading tokenizer from {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Loading model from {MODEL_NAME}...")


bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

model.eval()

num_layers = len(model.model.layers)
print(f"Loaded Total layers: {num_layers}")

import json
import pandas as pd
import numpy as np
import torch
import pickle
import random
from pathlib import Path
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig


DATA_PATH = Path("/content/commands_tangled_bow.json")
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 512
BATCH_SIZE = 4
SEED = 42
POOLING_METHOD = 'mean'
TUNE_REGULARIZATION = True
PERMUTATION_TEST_N = 50

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def load_json_or_jsonl(path: Path):
    records = []
    text = path.read_text(encoding="utf-8").strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            records = parsed
        elif isinstance(parsed, dict):
            records = [parsed]
    except json.JSONDecodeError:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

records = load_json_or_jsonl(DATA_PATH)

df = pd.DataFrame(records)
if 'command' in df.columns:
    df['full_text'] = df['command'].astype(str)
elif 'messages' in df.columns:
    df['full_text'] = df['messages'].fillna('').astype(str)
if 'label' not in df.columns:
    raise SystemExit("Dataset must have a 'label' field (0/1).")
df['label'] = df['label'].astype(int)

print(f"\nLoaded dataset: {len(df)} examples")
print(f"Positive examples: {df['label'].sum()}")
print(f"Negative examples: {(df['label']==0).sum()}")


train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=SEED)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=SEED)
#print(f"Training set: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")



tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

num_layers = len(model.model.layers)
print(f"\nModel has {num_layers} transformer layers")


early_layers = [0, 1]
middle_layers = [7, 11, 15, 19]
last_layers = [25, 26, 27] #0indexed
strategies = {f"Layer {i}": [i] for i in early_layers + middle_layers + last_layers}

print("\nLayer strategies:")
for k, v in strategies.items():
    print(f"  {k}: {v}")


#Append to command, slight increase
def format_command_with_prompt(command):
    return f"Question: Does this shell command access the internet?\nCommand: {command}\nAnswer:"

def extract_activations_from_layer(texts, layer_idx, batch_size=BATCH_SIZE, pooling='mean'):
    activations = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Layer {layer_idx}", leave=False):
        batch_texts = [format_command_with_prompt(t) for t in texts[i:i+batch_size]]
        inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, max_length=MAX_LENGTH, padding=True).to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.hidden_states
            layer_hidden = hidden_states[layer_idx]
            attention_mask = inputs['attention_mask']
            if pooling == 'mean':
                mask_exp = attention_mask.unsqueeze(-1).expand(layer_hidden.size()).float()
                summed = torch.sum(layer_hidden * mask_exp, 1)
                pooled = summed / attention_mask.sum(1, keepdim=True)
            else:
                final_idx = attention_mask.sum(dim=1) - 1
                pooled = torch.stack([layer_hidden[j, final_idx[j], :] for j in range(len(batch_texts))])
            activations.append(pooled.cpu().float().numpy())
        del outputs, hidden_states
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if not activations:
        return np.zeros((0, model.config.hidden_size), dtype=np.float32)
    return np.vstack(activations)


print("\nRunning hidden state diagnostics (mean/var per layer)...")
test_input = tokenizer("diagnostic check sentence", return_tensors="pt").to(DEVICE)
with torch.no_grad():
    diag_out = model(**test_input, output_hidden_states=True)
means = [h.mean().item() for h in diag_out.hidden_states]
vars_ = [h.var().item() for h in diag_out.hidden_states]
for i, (m, v) in enumerate(zip(means, vars_)):
    print(f"Layer {i:02d}: mean={m:.4f}, var={v:.4f}")


def permutation_test(X, y, clf_class, n=PERMUTATION_TEST_N):
    real_clf = clf_class().fit(X, y)
    real_f1 = f1_score(y, real_clf.predict(X))
    rand_f1s = []
    for _ in range(n):
        y_r = np.random.permutation(y)
        c = clf_class().fit(X, y_r)
        rand_f1s.append(f1_score(y_r, c.predict(X)))
    pval = (sum(r >= real_f1 for r in rand_f1s) + 1) / (n + 1)
    return real_f1, np.mean(rand_f1s), np.std(rand_f1s), pval


results = {}
for strategy_name, layer_indices in strategies.items():
    layer = layer_indices[0]
    print(f"\n=== {strategy_name} (Layer {layer}) ===")

    X_train = extract_activations_from_layer(train_df['full_text'].tolist(), layer, pooling=POOLING_METHOD)
    X_val = extract_activations_from_layer(val_df['full_text'].tolist(), layer, pooling=POOLING_METHOD)
    X_test = extract_activations_from_layer(test_df['full_text'].tolist(), layer, pooling=POOLING_METHOD)


    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)


    if TUNE_REGULARIZATION:
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
        clf_cv = GridSearchCV(LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
                              param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
        clf_cv.fit(X_train_s, train_df['label'])
        clf = clf_cv.best_estimator_
        print(f"  Tuned C={clf_cv.best_params_['C']}, CV AUC={clf_cv.best_score_:.4f}")
    else:
        clf = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED).fit(X_train_s, train_df['label'])

    for split, (X, y_true) in {'Train': (X_train_s, train_df['label']),
                               'Val': (X_val_s, val_df['label']),
                               'Test': (X_test_s, test_df['label'])}.items():
        preds = clf.predict(X)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        try:
            auc = roc_auc_score(y_true, clf.predict_proba(X)[:, 1])
        except Exception:
            auc = float('nan')
        print(f"  {split}: Acc={acc:.3f}, F1={f1:.3f}, AUC={auc:.3f}")


    results[strategy_name] = {
        'layer': layer,
        'clf': clf,
        'scaler': scaler
    }

print("\nAll probes completed.")



