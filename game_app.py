"""
app.py
秘封倶楽部 瞬間ベクトル推薦システム のStreamlit UI

タブ構成:
- 「通常モード」: 好きな瞬間から似た瞬間を探す(従来機能)
- 「ゲームモード」: 出題者が指定した曲を、回答者が別の曲・瞬間から誘導するゲーム
  正解条件: 選んだ瞬間で検索した結果の上位10位以内に、出題曲のいずれかの
  区間が含まれていれば正解。
  なぜexclude_same_trackを使わないか: 通常モードは「他の曲の似た瞬間を探す」
  ことが目的なので出題曲(=クエリと同じ曲)を除外するが、ゲームモードでは
  逆に「出題曲が結果に出てくるかどうか」自体を判定したいため、除外しない。
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
    return load_search_assets(ROOT)


@st.cache_data
def get_audio_for_track(album: str, track: str) -> np.ndarray:
    flac_path = ROOT / "music" / album / f"{track}.flac"
    return load_audio(flac_path)


def extract_clip(audio: np.ndarray, start_sec: float, duration_sec: float = 10.0) -> np.ndarray:
    start_sample = int(start_sec * SR)
    end_sample = int((start_sec + duration_sec) * SR)
    return audio[start_sample:end_sample]


def audio_to_bytes(clip: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, clip, SR, format="WAV")
    buffer.seek(0)
    return buffer.read()


def get_track_options(metadata):
    """曲一覧をmetadataから重複なく抽出し、選択用のラベルと対応表を返す。"""
    tracks = sorted(set((m["album"], m["track"]) for m in metadata))
    labels = [f"{album} / {track}" for album, track in tracks]
    return tracks, labels


def render_normal_mode(index, embeddings, metadata):
    """従来の「好きな瞬間から似た瞬間を探す」機能。"""
    tracks, track_labels = get_track_options(metadata)

    selected_label = st.selectbox("曲を選ぶ", track_labels, key="normal_track")
    selected_album, selected_track = tracks[track_labels.index(selected_label)]

    available_starts = sorted(
        m["start_sec"] for m in metadata
        if m["album"] == selected_album and m["track"] == selected_track
    )

    selected_start = st.select_slider(
        "好きな瞬間(開始秒)を選ぶ",
        options=available_starts,
        format_func=lambda s: f"{s:.0f}秒〜{s + 10:.0f}秒",
        key="normal_start",
    )

    query_audio = get_audio_for_track(selected_album, selected_track)
    query_clip = extract_clip(query_audio, selected_start)
    st.write("**選んだ瞬間:**")
    st.audio(audio_to_bytes(query_clip), format="audio/wav")

    top_k = st.slider("表示件数", min_value=3, max_value=20, value=10, key="normal_topk")

    if st.button("似た瞬間を探す", type="primary", key="normal_search_btn"):
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


def render_game_mode(index, embeddings, metadata):
    """
    ゲームモード:
    出題者が口頭で指定した曲を、回答者が別の曲・瞬間から誘導する。
    回答者は自分の予想する曲・瞬間を選んで検索し、
    出題曲が上位10位以内に入っているかを確認する。
    """
    st.write("出題者から伝えられた曲を思い浮かべながら、それが推薦の上位に来そうな曲・瞬間を選んでみてください。")

    tracks, track_labels = get_track_options(metadata)

    selected_label = st.selectbox("あなたが選ぶ曲", track_labels, key="game_track")
    selected_album, selected_track = tracks[track_labels.index(selected_label)]

    available_starts = sorted(
        m["start_sec"] for m in metadata
        if m["album"] == selected_album and m["track"] == selected_track
    )

    selected_start = st.select_slider(
        "選ぶ瞬間(開始秒)",
        options=available_starts,
        format_func=lambda s: f"{s:.0f}秒〜{s + 10:.0f}秒",
        key="game_start",
    )

    query_audio = get_audio_for_track(selected_album, selected_track)
    query_clip = extract_clip(query_audio, selected_start)
    st.write("**選んだ瞬間:**")
    st.audio(audio_to_bytes(query_clip), format="audio/wav")

    st.write("---")
    target_track_label = st.selectbox(
        "出題された曲はどれ？(結果と照らし合わせるため選択)",
        track_labels,
        key="game_target_track",
    )
    target_album, target_track = tracks[track_labels.index(target_track_label)]

    if st.button("誘導できたか確認する", type="primary", key="game_search_btn"):
        query_vec = query_by_existing_segment(embeddings, metadata, selected_album, selected_track, selected_start
        )
        # 回答者が選んだ曲自身の別区間が上位を占めてしまい、
        # 出題曲が押し出される問題への対応として、選んだ曲自身は除外する。
        # (出題曲は除外対象にしていないので、そのまま結果に残る)
        results = search_similar(index, metadata, query_vec, top_k=10, exclude_same_track=selected_track)

        # 出題曲が結果内の何位にあるかを探す
        hit_rank = None
        for i, r in enumerate(results, 1):
            if r["album"] == target_album and r["track"] == target_track:
                hit_rank = i
                break

        if hit_rank:
            st.success(f"🎉 正解！ 出題曲が {hit_rank}位 にランクインしました")
        else:
            st.error("😢 惜しい、出題曲は上位10位以内に入りませんでした")

        st.write("### 検索結果 (top 10)")
        for i, r in enumerate(results, 1):
            marker = "👉 " if (r["album"] == target_album and r["track"] == target_track) else ""
            st.write(f"{marker}**{i}. {r['album']} / {r['track']}** ({r['start_sec']:.0f}秒〜) — 類似度: {r['score']:.3f}")
            result_audio = get_audio_for_track(r["album"], r["track"])
            result_clip = extract_clip(result_audio, r["start_sec"])
            st.audio(audio_to_bytes(result_clip), format="audio/wav")


def main():
    st.title("秘封倶楽部 瞬間ベクトル推薦システム")
    st.caption("好きな10秒に近い、別の曲の瞬間を探す")

    index, embeddings, metadata = get_assets()

    tab_normal, tab_game = st.tabs(["通常モード", "ゲームモード"])

    with tab_normal:
        render_normal_mode(index, embeddings, metadata)

    with tab_game:
        render_game_mode(index, embeddings, metadata)


if __name__ == "__main__":
    main()