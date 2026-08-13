"""
app.py
秘封倶楽部 瞬間ベクトル推薦システム のStreamlit UI

フロー:
1. 曲(track)を選ぶ
2. その曲内の区間(start_sec)を選ぶ → 「好きな瞬間」の指定
3. 検索ボタンで類似セグメントを検索(同一曲は自動除外)
4. 結果を一覧表示、各セグメントをその場で再生して確認できる

なぜこの構成か:
- 音源ファイルへのフルパスがあれば、soundfileで該当区間だけを
  その場で切り出してst.audio()に渡せるため、
  事前に全セグメントを音声ファイルとして書き出しておく必要がない
  (ディスク容量と前処理時間の節約)。
"""

import io
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st

from search import load_search_assets, query_by_existing_segment, search_similar
from segment import load_audio, SR


ROOT = Path(__file__).parent.parent


@st.cache_resource
def get_assets():
    """
    FAISSインデックス・embeddings・metadataは重いので、
    Streamlitのキャッシュ機構で一度だけロードする。
    (毎回の操作でリロードすると体感速度が悪くなるため)
    """
    return load_search_assets(ROOT)


@st.cache_data
def get_audio_for_track(album: str, track: str) -> np.ndarray:
    """
    曲全体の波形をキャッシュ付きでロードする。
    同じ曲の別区間を何度も試す場合に毎回ファイル読み込みが
    走らないようにするため。
    """
    flac_path = ROOT / "music" / album / f"{track}.flac"
    return load_audio(flac_path)


def extract_clip(audio: np.ndarray, start_sec: float, duration_sec: float = 10.0) -> np.ndarray:
    """波形から指定区間を切り出す。"""
    start_sample = int(start_sec * SR)
    end_sample = int((start_sec + duration_sec) * SR)
    return audio[start_sample:end_sample]


def audio_to_bytes(clip: np.ndarray) -> bytes:
    """
    numpy配列をst.audio()が受け取れるバイト列(WAV形式)に変換する。
    ファイルとして書き出さずメモリ上で完結させるため、
    io.BytesIOを使っている。
    """
    buffer = io.BytesIO()
    sf.write(buffer, clip, SR, format="WAV")
    buffer.seek(0)
    return buffer.read()


def main():
    st.title("秘封倶楽部 瞬間ベクトル推薦システム")
    st.caption("好きな10秒に近い、別の曲の瞬間を探す")

    index, embeddings, metadata = get_assets()

    # 曲一覧をmetadataから重複なく抽出
    tracks = sorted(set((m["album"], m["track"]) for m in metadata))
    track_labels = [f"{album} / {track}" for album, track in tracks]

    selected_label = st.selectbox("曲を選ぶ", track_labels)
    selected_album, selected_track = tracks[track_labels.index(selected_label)]

    # 選んだ曲の中で存在するstart_secの一覧(セグメント境界)を抽出
    available_starts = sorted(
        m["start_sec"] for m in metadata
        if m["album"] == selected_album and m["track"] == selected_track
    )

    selected_start = st.select_slider(
        "好きな瞬間(開始秒)を選ぶ",
        options=available_starts,
        format_func=lambda s: f"{s:.0f}秒〜{s + 10:.0f}秒",
    )

    # クエリ区間をその場で再生できるようにする
    query_audio = get_audio_for_track(selected_album, selected_track)
    query_clip = extract_clip(query_audio, selected_start)
    st.write("**選んだ瞬間:**")
    st.audio(audio_to_bytes(query_clip), format="audio/wav")

    top_k = st.slider("表示件数", min_value=3, max_value=20, value=10)

    if st.button("似た瞬間を探す", type="primary"):
        query_vec = query_by_existing_segment(
            embeddings, metadata, selected_album, selected_track, selected_start
        )
        results = search_similar(
            index, metadata, query_vec, top_k=top_k, exclude_same_track=selected_track
        )

        st.write(f"### 検索結果 ({len(results)}件)")

        for i, r in enumerate(results, 1):
            st.write(f"**{i}. {r['album']} / {r['track']}** ({r['start_sec']:.0f}秒〜) — 類似度: {r['score']:.3f}")
            result_audio = get_audio_for_track(r["album"], r["track"])
            result_clip = extract_clip(result_audio, r["start_sec"])
            st.audio(audio_to_bytes(result_clip), format="audio/wav")


if __name__ == "__main__":
    main()