"""
search.py
「好きな瞬間」に近いセグメントをFAISSで検索する。

対応するクエリの与え方:
1. 既存曲の「何秒目」を指定 (query_by_existing_segment)
   - embeddings.pkl内のセグメントをそのままクエリベクトルとして使う
2. 新しい音声ファイルを渡す (query_by_new_audio)
   - segment.py/embed.pyと同じ処理でその場でembedding化してクエリにする

どちらも最終的には「クエリベクトル(1本)」を得てから
search_similar()に渡す、という共通の流れにしている。
"""

import pickle
from pathlib import Path

import faiss
import numpy as np
import torch

from segment import load_audio, SR
from embed import get_device, load_clap


def normalize_vector(vec: np.ndarray) -> np.ndarray:
    """1本のベクトルをL2正規化する(build_index.pyのnormalizeと同じロジック)。"""
    norm = np.linalg.norm(vec)
    return vec / norm


def load_search_assets(root: Path):
    """
    検索に必要な3点セットをロードする:
    - FAISSインデックス
    - embeddings(クエリ用途で既存セグメントを参照する場合に使う)
    - metadata(検索結果を人間が読める形に変換するため)
    """
    index = faiss.read_index(str(root / "hifuu_index.faiss"))

    with open(root / "embeddings.pkl", "rb") as f:
        data = pickle.load(f)

    return index, data["embeddings"], data["metadata"]


def query_by_existing_segment(embeddings: np.ndarray, metadata: list[dict], album: str, track: str, start_sec: float) -> np.ndarray:
    """
    既存曲の「何秒目」からクエリベクトルを取り出す。
    """
    for i, m in enumerate(metadata):
        if m["album"] == album and m["track"] == track and m["start_sec"] == start_sec:
            return embeddings[i].astype(np.float32)

    raise ValueError(
        f"該当セグメントが見つかりません: album={album}, track={track}, start_sec={start_sec}\n"
        f"start_secはsegment.pyのHOP_SEC刻み(0, 5, 10, ...)である必要があります。"
    )


def query_by_new_audio(filepath: Path) -> np.ndarray:
    """
    新規音声ファイル(10秒切り出し済み想定)をCLAPでembedding化してクエリにする。
    embed.pyと同じCLAPモデルを使うことで、既存セグメントと同じベクトル空間に
    正しく埋め込まれることを保証する。
    """
    device = get_device()
    processor, model = load_clap(device)

    audio = load_audio(filepath)  # segment.pyのload_audioを流用、SR=48000で統一済み
    inputs = processor(audio=[audio], sampling_rate=SR, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        audio_embed = model.get_audio_features(**inputs)

    if not torch.is_tensor(audio_embed):
        if hasattr(audio_embed, "pooler_output"):
            audio_embed = audio_embed.pooler_output
        elif hasattr(audio_embed, "last_hidden_state"):
            audio_embed = audio_embed.last_hidden_state[:, 0, :]
        else:
            raise TypeError(f"想定外の戻り値型: {type(audio_embed)}")

    return audio_embed.cpu().numpy()[0].astype(np.float32)


def search_similar(index: faiss.Index, metadata: list[dict], query_vec: np.ndarray, top_k: int = 10, exclude_self: bool = True, exclude_same_track: str = None):
    """
    クエリベクトルに近いセグメントをtop_k件返す。
    exclude_self=Trueの場合、類似度がほぼ1.0(=自分自身)の結果を除外する。
    exclude_same_trackにtrack名を渡すと、同じ曲のセグメントを全て除外する。
    なぜ必要か: CLAPは同じ曲内では音色・キー・テンポが共通するため、
    「他の曲の似た瞬間を探す」というコンセプト上、同じ曲は比較対象として
    そもそも意味を持たないことが多い。
    """
    query_vec = normalize_vector(query_vec).reshape(1, -1)

    # 除外条件がある分、多めに候補を取得しておく
    search_k = min(top_k * 5, index.ntotal)
    scores, indices = index.search(query_vec, search_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if exclude_self and score > 0.999:
            continue
        m = metadata[idx]
        if exclude_same_track is not None and m["track"] == exclude_same_track:
            continue
        results.append({
            "album": m["album"],
            "track": m["track"],
            "start_sec": m["start_sec"],
            "end_sec": m["end_sec"],
            "score": float(score),
        })
        if len(results) >= top_k:
            break

    return results


def print_results(results: list[dict]):
    print(f"\n{'順位':<4} {'類似度':<8} アルバム/曲名 (開始秒)")
    print("-" * 60)
    for i, r in enumerate(results, 1):
        print(f"{i:<4} {r['score']:.4f}  {r['album']}/{r['track']} ({r['start_sec']:.0f}秒〜)")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    index, embeddings, metadata = load_search_assets(root)
 
    example_album = metadata[0]["album"]
    example_track = metadata[0]["track"]
    example_start = metadata[0]["start_sec"]
 
    print(f"クエリ: {example_album}/{example_track} ({example_start}秒〜)")
    query_vec = query_by_existing_segment(embeddings, metadata, example_album, example_track, example_start)
    results = search_similar(index, metadata, query_vec, top_k=10, exclude_same_track=example_track)
    print_results(results)
 
    # --- 使用例2: 新規音声ファイルを指定する場合(コメントアウト) ---
    # query_vec = query_by_new_audio(Path("my_favorite_moment.flac"))
    # results = search_similar(index, metadata, query_vec, top_k=10, exclude_self=False)
    # print_results(results)
