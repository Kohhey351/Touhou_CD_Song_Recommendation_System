import pickle
from pathlib import Path

root = Path(__file__).parent.parent
with open(root / "embeddings.pkl", "rb") as f:
    data = pickle.load(f)

print("embeddings shape:", data["embeddings"].shape)
print("num metadata entries:", len(data["metadata"]))
print("sample metadata:", data["metadata"][:3])
