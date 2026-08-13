"""
segment.py
音源ファイル(FLAC)を10秒単位のセグメントに分割する。
5秒刻みでオーバーラップさせるオプション付き。

なぜオーバーラップさせるか:
- 境界で「好きなフレーズ」が分断されてしまうのを避けるため。
  例えば8秒目にサビの盛り上がりがあると、0-10秒と10-20秒の
  どちらのセグメントにも「中途半端に」しか含まれない。
  5秒ずらしたセグメント(5-15秒)を追加で持つことで、
  サビ全体を捉えたセグメントが生まれる確率が上がる。
"""

import os
from pathlib import Path
from dataclasses import dataclass

import librosa
import numpy as np


SEGMENT_SEC = 10.0
HOP_SEC = 5.0  # オーバーラップさせる場合のずらし幅。SEGMENT_SECと同じ値にすればオーバーラップなし。
SR = 48000     # CD音質のサンプルレート。CLAP側で48000のSRが必要だったので。


@dataclass
class Segment:
    album: str
    track: str
    start_sec: float
    end_sec: float
    audio: np.ndarray  # 波形データ(モノラル)


def load_audio(filepath: Path) -> np.ndarray:
    """
    FLACを読み込んでモノラル波形を返す。
    なぜモノラルにするか: CLAP等の音響embeddingモデルは
    基本的にモノラル入力を前提としているものが多いため。
    """
    audio, _ = librosa.load(filepath, sr=SR, mono=True)
    return audio


def split_into_segments(audio: np.ndarray, album: str, track: str) -> list[Segment]:
    """
    波形をSEGMENT_SEC単位・HOP_SEC刻みで分割する。
    最後の端数(SEGMENT_SEC未満の余り)は捨てる方針。
    理由: 短すぎるセグメントはembeddingの意味的な精度が落ちるため、
    無理に含めるより除外した方が検索結果の質が安定する。
    """
    segment_samples = int(SEGMENT_SEC * SR)
    hop_samples = int(HOP_SEC * SR)

    segments = []
    start_sample = 0
    while start_sample + segment_samples <= len(audio):
        chunk = audio[start_sample : start_sample + segment_samples]
        start_sec = start_sample / SR
        end_sec = (start_sample + segment_samples) / SR
        segments.append(Segment(album, track, start_sec, end_sec, chunk))
        start_sample += hop_samples

    return segments


def process_album_dir(album_dir: Path) -> list[Segment]:
    """
    1アルバムフォルダ内の全FLACファイルをセグメント分割する。
    ファイル名の拡張子は.flac固定を想定(XLDの出力フォーマットに合わせている)。
    """
    all_segments = []
    for filepath in sorted(album_dir.glob("*.flac")):
        track_name = filepath.stem
        audio = load_audio(filepath)
        segments = split_into_segments(audio, album_dir.name, track_name)
        all_segments.extend(segments)
        print(f"[{album_dir.name}/{track_name}] {len(segments)} segments")

    return all_segments


def process_music_dir(music_dir: Path) -> list[Segment]:
    """
    music_dir配下の全アルバムフォルダを走査し、まとめてセグメント分割する。
    music_dir/album1/track01.flac のような構成を想定。
    """
    all_segments = []
    for album_dir in sorted(music_dir.iterdir()):
        if not album_dir.is_dir():
            continue  # フォルダ以外(誤って紛れ込んだファイル等)はスキップ
        segments = process_album_dir(album_dir)
        all_segments.extend(segments)

    return all_segments


if __name__ == "__main__":
    # 実行例: hifuu_recsys直下から python src/segment.py
    root = Path(__file__).parent.parent
    music_dir = root / "music"

    segments = process_music_dir(music_dir)
    print(f"\n合計 {len(segments)} セグメント")