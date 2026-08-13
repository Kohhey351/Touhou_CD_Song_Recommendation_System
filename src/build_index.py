"""
build_index.py
embeddings.pklからFAISSインデックスを構築する。

なぜコサイン類似度を使うか:
- 「音の大きさ」ではなく「音響的な方向性(意味的な近さ)」を
  比較したいため、ベクトルの向きだけを見るコサイン類似度が適切。
- FAISSでコサイン類似度を使うには、ベクトルを正規化した上で
  内積(Inner Product)インデックスを使うのが標準的な方法。
"""

import pickle
from pathlib import Path

import faiss
import numpy as np


def normalize(embeddings: np.ndarray) -> np.ndarray:
    """
    各ベクトルをL2正規化する(長さ1にする)。
    正規化後の内積 = コサイン類似度になるため、
    これによって「内積検索」がそのまま「コサイン類似度検索」になる。
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    正規化済みembeddingからFlatな(近似なしの)内積インデックスを作る。
    なぜFlat(総当たり)でいいか: 540セグメント程度の規模なら
    近似インデックス(IVF等)を使う必要はなく、Flatで十分高速かつ正確。
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product = 正規化済みならコサイン類似度
    index.add(embeddings)
    return index


def build_and_save(embeddings_path: Path, index_path: Path):
    with open(embeddings_path, "rb") as f:
        data = pickle.load(f)

    embeddings = data["embeddings"].astype(np.float32)  # FAISSはfloat32を要求する
    embeddings = normalize(embeddings)

    index = build_faiss_index(embeddings)

    faiss.write_index(index, str(index_path))
    print(f"FAISSインデックスを保存: {index_path}")
    print(f"登録ベクトル数: {index.ntotal}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    embeddings_path = root / "embeddings.pkl"
    index_path = root / "hifuu_index.faiss"

    build_and_save(embeddings_path, index_path)
