import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import login

# ============================================================
# LOGIN TO HUGGINGFACE (Required for Gemma)
# ============================================================
login("your token")  # your token

# ============================================================
# CONFIG
# ============================================================
CSV_PATH = r"Dataset_Phishdump.csv"
SAVE_PATH = r"embeddings.npy"
MODEL_NAME = "google/gemma-2b-it"

BATCH_SIZE = 64
MAX_LENGTH = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", DEVICE)

# ============================================================
# 1. Load Dataset
# ============================================================
df = pd.read_csv(CSV_PATH, header=None)
df = df.drop(columns=[0])
df.columns = ["text", "label"]

texts = df["text"].astype(str).tolist()
labels = df["label"].values

print(f"Loaded {len(texts)} samples.")
print("Example text:", texts[0])

# ============================================================
# 2. Load Model + Tokenizer
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True
).to(DEVICE)

model.eval()
print("Model loaded:", MODEL_NAME)

# ============================================================
# 3. Embedding Function
# ============================================================
@torch.inference_mode()
def get_embeddings(batch_texts):
    tokens = tokenizer(
        batch_texts,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    outputs = model(**tokens)
    emb = outputs.last_hidden_state.mean(dim=1)
    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

    return emb.cpu().float().numpy()

# ============================================================
# 4. Extraction
# ============================================================
all_embeddings = []

for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="Extracting embeddings..."):
    batch = texts[i:i+BATCH_SIZE]
    emb = get_embeddings(batch)
    all_embeddings.append(emb)

X = np.vstack(all_embeddings)
print("Final embedding matrix shape:", X.shape)

# ============================================================
# 5. Save
# ============================================================
np.save(SAVE_PATH, X)
print(f"Embeddings saved to: {SAVE_PATH}")

# ============================================================
# 6. Validation
# ============================================================
unique = len({tuple(v) for v in X})
print("Unique embeddings:", unique)
print("Duplicated embeddings:", len(X) - unique)

assert X.shape[0] == len(df), "Embedding alignment ERROR!"
print("✔ Embeddings correctly aligned with CSV rows.")
