"""
指定したalbum/track/start_secのセグメントを実際に音声ファイルとして
切り出して保存する。耳で聞いてフレーズが正しく捉えられているか確認するため。
"""
import soundfile as sf
from pathlib import Path
from segment import load_audio, SR

def extract_and_save(flac_path: Path, start_sec: float, duration_sec: float, output_path: Path):
    audio = load_audio(flac_path)
    start_sample = int(start_sec * SR)
    end_sample = int((start_sec + duration_sec) * SR)
    clip = audio[start_sample:end_sample]
    sf.write(output_path, clip, SR)
    print(f"保存: {output_path}")

if __name__ == "__main__":
    root = Path(__file__).parent.parent
    flac_path = root / "ozoramajutsu" / "01 月面ツアーへようこそ 1.flac"
    extract_and_save(flac_path, start_sec=10.0, duration_sec=10.0, output_path=root / "check_clip.wav")