"""
embed.py
segment.pyで作った10秒セグメントをCLAPでベクトル化する。

なぜCLAPか:
- 音楽・音声の「意味的な近さ」を捉えるのに向いている
- 10秒程度の入力長と相性がいい
- transformers経由で簡単に扱える

出力:
- 各セグメントごとに1本のembeddingベクトル(numpy配列)
- メタデータ(album, track, start_sec, end_sec)と対応付けて保存
"""

import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from segment import process_music_dir, SR, Segment


MODEL_NAME = "laion/larger_clap_music"


def get_device() -> torch.device:
    """
    Apple SiliconならMPS、なければCPUを使う。
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_clap(device: torch.device):
    """
    CLAPモデルとプロセッサをロードする。
    初回実行時はHugging Faceからモデルをダウンロードするため
    数分かかることがある(2回目以降はキャッシュされて速い)。
    """
    processor = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return processor, model


def embed_segments(
    segments: list[Segment], processor, model, device: torch.device, batch_size: int = 8
) -> np.ndarray:
    """
    セグメントのリストをバッチ処理でembeddingに変換する。
    なぜバッチ処理か: 1本ずつ処理するより効率的で、
    数十曲規模でも待ち時間を短縮できるため。
    """
    all_embeddings = []

    for i in range(0, len(segments), batch_size):
        batch = segments[i : i + batch_size]
        audios = [seg.audio for seg in batch]

        inputs = processor(audio=audios, sampling_rate=SR, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            audio_embed = model.get_audio_features(**inputs)
        
        # transformersのバージョンによって戻り値がテンソル直接の場合と、
        # BaseModelOutputWithPoolingのようなオブジェクトの場合がある。
        if not torch.is_tensor(audio_embed):
            if hasattr(audio_embed, "pooler_output"):
                audio_embed = audio_embed.pooler_output
            elif hasattr(audio_embed, "last_hidden_state"):
                audio_embed = audio_embed.last_hidden_state[:, 0, :]
            else:
                raise TypeError(f"想定外の戻り値型: {type(audio_embed)}")

        all_embeddings.append(audio_embed.cpu().numpy())
        print(f"  embedded {min(i + batch_size, len(segments))}/{len(segments)}")

    return np.concatenate(all_embeddings, axis=0)


def build_embedding_store(album_dir: Path, output_path: Path):
    """
    アルバムフォルダ全体を処理し、embeddingとメタデータをpickleで保存する。
    pickleを使う理由: numpy配列とdataclassのリストを一緒に保存でき、
    FAISSインデックス構築時にそのままロードして使えるため。
    """
    device = get_device()
    print(f"using device: {device}")

    processor, model = load_clap(device)

    segments = process_music_dir(album_dir)
    print(f"\n{len(segments)} segments を embedding します")

    embeddings = embed_segments(segments, processor, model, device)

    # audioデータ自体は保存しない(容量が大きくなるため)。
    # メタデータのみ抽出して別途保存する。
    metadata = [
        {"album": seg.album, "track": seg.track, "start_sec": seg.start_sec, "end_sec": seg.end_sec}
        for seg in segments
    ]

    with open(output_path, "wb") as f:
        pickle.dump({"embeddings": embeddings, "metadata": metadata}, f)

    print(f"\n保存完了: {output_path}")
    print(f"embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    root = Path(__file__).parent.parent
    music_dir = root / "music"
    output_path = root / "embeddings.pkl"

    build_embedding_store(music_dir, output_path)