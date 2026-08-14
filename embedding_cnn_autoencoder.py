"""音楽断片の音響埋め込みを学習し、類似検索・推薦に使うCNN AutoEncoder。

  - librosaのlog-melスペクトログラムを入力にする
  - 自作CNN Encoderで各10秒断片を128次元の埋め込み z に圧縮する
  - Decoderで入力を再構成する教師なし学習を行う
  - 曲単位でtrain/validation/testを分離し、情報漏洩を避ける
  - テスト再構成誤差と、同一曲断片検索のPrecision@K/MRRを評価する

出力:
  outputs/autoencoder.pth       学習済み重み
  outputs/embeddings.npy        全断片の128次元埋め込み
  outputs/metadata.csv          埋め込みIDと曲名・区間
  outputs/reconstructions.png   テスト曲の入出力比較
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ---------- 設定 ----------
AUDIO_DIR = Path("./audio")       # 曲ファイルを置くフォルダ
OUTPUT_DIR = Path("./outputs")
SAMPLE_RATE = 22_050
SEGMENT_SECONDS = 10
N_FFT = 2_048
HOP_LENGTH = 512
N_MELS = 128
T_FIXED = 431                      # 10秒を上記設定で変換したときの時間フレーム数
EMBED_DIM = 128
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-3
SEED = 42


@dataclass(frozen=True)
class Segment:
    song_name: str
    start_sec: int
    feature_path: Path


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def logmel_from_audio(y: np.ndarray) -> np.ndarray:
    """10秒波形を、0--1に正規化した [128, 431] log-mel特徴量へ変換する。"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    # 元ファイルの設計を踏襲: 各断片のスペクトル形状を0--1へ正規化する。
    log_mel = librosa.power_to_db(mel, ref=np.max)
    log_mel = (log_mel - log_mel.min()) / (log_mel.max() - log_mel.min() + 1e-8)
    if log_mel.shape[1] >= T_FIXED:
        return log_mel[:, :T_FIXED].astype(np.float32)
    return np.pad(log_mel, ((0, 0), (0, T_FIXED - log_mel.shape[1]))).astype(np.float32)


def build_feature_cache(audio_dir: Path, cache_dir: Path) -> list[Segment]:
    """曲を10秒ごとに分割してlog-melを.npzとして保存し、メタデータを返す。"""
    extensions = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_paths = sorted(p for p in audio_dir.rglob("*") if p.suffix.lower() in extensions)
    if len(audio_paths) < 5:
        raise ValueError(f"{audio_dir} に5曲以上置いてください。見つかった曲数: {len(audio_paths)}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    segments: list[Segment] = []
    samples_per_segment = SAMPLE_RATE * SEGMENT_SECONDS
    for audio_path in audio_paths:
        # 同名の別拡張子を区別するため、相対パスをファイル名に含める。
        song_name = "__".join(audio_path.relative_to(audio_dir).with_suffix("").parts)
        y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        segment_count = len(y) // samples_per_segment  # 10秒未満の末尾は除外する
        for index in range(segment_count):
            start = index * samples_per_segment
            feature = logmel_from_audio(y[start:start + samples_per_segment])
            feature_path = cache_dir / f"{song_name}__{index:03d}.npy"
            np.save(feature_path, feature)
            segments.append(Segment(song_name, index * SEGMENT_SECONDS, feature_path))
        print(f"{song_name}: {segment_count} segments")
    if not segments:
        raise ValueError("10秒以上の音声断片がありません。")
    return segments


class FeatureDataset(Dataset):
    """AutoEncoder用。入力を正解として返すためラベルは不要。"""

    def __init__(self, segments: list[Segment]):
        self.segments = segments

    def __len__(self) -> int:
        return len(self.segments)

    def __getitem__(self, index: int) -> torch.Tensor:
        feature = np.load(self.segments[index].feature_path)
        return torch.from_numpy(feature).unsqueeze(0)  # [1, 128, 431]


class CNNEncoder(nn.Module):
    """log-mel画像を128次元の音響埋め込みへ圧縮する。"""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.global_average_pool = nn.AdaptiveAvgPool2d(1)
        self.to_embedding = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.global_average_pool(h).flatten(1)
        return self.to_embedding(h)


class CNNDecoder(nn.Module):
    """128次元埋め込みから固定サイズのlog-mel画像を復元する。"""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.base_height, self.base_width = N_MELS // 4, T_FIXED // 4
        self.from_embedding = nn.Linear(embed_dim, 64 * self.base_height * self.base_width)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_embedding(z).view(-1, 64, self.base_height, self.base_width)
        reconstruction = self.deconv(h)
        # 431は4で割り切れないため、最後に幅を厳密に合わせる。
        return nn.functional.interpolate(reconstruction, size=(N_MELS, T_FIXED), mode="bilinear", align_corners=False)


class MusicAutoEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim)
        self.decoder = CNNDecoder(embed_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def split_by_song(segments: list[Segment]) -> tuple[list[Segment], list[Segment], list[Segment]]:
    """同一曲の断片が複数の分割に混ざらないよう、曲名で分離する。"""
    groups = np.array([segment.song_name for segment in segments])
    indices = np.arange(len(segments))
    if len(np.unique(groups)) < 5:
        raise ValueError("曲単位のtrain/validation/test分割には5曲以上必要です。")
    # 5曲時でもテスト集合に少なくとも2曲を含め、同曲／別曲の検索を比較できるようにする。
    first = GroupShuffleSplit(n_splits=1, test_size=0.40, random_state=SEED)
    train_validation_indices, test_indices = next(first.split(indices, groups=groups))
    train_validation_groups = groups[train_validation_indices]
    second = GroupShuffleSplit(n_splits=1, test_size=1 / 3, random_state=SEED + 1)
    train_relative, validation_relative = next(
        second.split(train_validation_indices, groups=train_validation_groups)
    )
    train = [segments[train_validation_indices[i]] for i in train_relative]
    validation = [segments[train_validation_indices[i]] for i in validation_relative]
    test = [segments[i] for i in test_indices]
    return train, validation, test


def reconstruction_loss(model: nn.Module, loader: DataLoader, device: torch.device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total = 0.0
    with torch.set_grad_enabled(training):
        for x in loader:
            x = x.to(device)
            reconstruction, _ = model(x)
            loss = nn.functional.mse_loss(reconstruction, x)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def embed_segments(model: MusicAutoEncoder, segments: list[Segment], device: torch.device) -> tuple[np.ndarray, pd.DataFrame]:
    model.eval()
    vectors: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for embedding_id, segment in enumerate(segments):
        x = torch.from_numpy(np.load(segment.feature_path)).unsqueeze(0).unsqueeze(0).to(device)
        _, z = model(x)
        vectors.append(z.squeeze(0).cpu().numpy())
        records.append({"embedding_id": embedding_id, "song_name": segment.song_name,
                        "start_sec": segment.start_sec, "end_sec": segment.start_sec + SEGMENT_SECONDS})
    return np.stack(vectors), pd.DataFrame(records)


def l2_normalize(vectors: np.ndarray, axis: int = -1) -> np.ndarray:
    return vectors / (np.linalg.norm(vectors, axis=axis, keepdims=True) + 1e-8)


def retrieval_metrics(embeddings: np.ndarray, metadata: pd.DataFrame, k: int = 5) -> tuple[float, float]:
    """同一曲の別断片を正解とするPrecision@KとMRR。テスト曲だけで測る。"""
    normalized = l2_normalize(embeddings)
    similarities = normalized @ normalized.T
    precisions, reciprocal_ranks = [], []
    names = metadata.song_name.to_numpy()
    for query in range(len(embeddings)):
        relevant = (names == names[query])
        relevant[query] = False
        if not relevant.any():
            continue
        ranking = np.argsort(-similarities[query])
        ranking = ranking[ranking != query]
        precisions.append(relevant[ranking[:k]].mean())
        first_relevant_rank = np.flatnonzero(relevant[ranking])[0] + 1
        reciprocal_ranks.append(1.0 / first_relevant_rank)
    if not precisions:
        raise ValueError("検索評価には、テスト集合内で各曲が2断片以上必要です。")
    return float(np.mean(precisions)), float(np.mean(reciprocal_ranks))


def recommend(embeddings: np.ndarray, metadata: pd.DataFrame, liked_ids: list[int], disliked_ids: list[int] | None = None, top_k: int = 5) -> pd.DataFrame:
    """好き／嫌い断片から好みベクトルを作り、近い未選択断片を返す。"""
    vectors = l2_normalize(embeddings)
    preference = vectors[liked_ids].mean(axis=0)
    if disliked_ids:
        preference -= 0.3 * vectors[disliked_ids].mean(axis=0)
    preference = l2_normalize(preference[None, :])[0]
    scores = vectors @ preference
    excluded = set(liked_ids) | set(disliked_ids or [])
    result = metadata.copy()
    result["similarity"] = scores
    return result[~result.embedding_id.isin(excluded)].sort_values("similarity", ascending=False).head(top_k)


@torch.no_grad()
def save_reconstruction_figure(model: MusicAutoEncoder, loader: DataLoader, device: torch.device) -> None:
    model.eval()
    x = next(iter(loader))[:3].to(device)
    reconstruction, _ = model(x)
    fig, axes = plt.subplots(2, len(x), figsize=(5 * len(x), 6), squeeze=False)
    for i in range(len(x)):
        for row, image, title in ((0, x[i, 0], "Original"), (1, reconstruction[i, 0], "Reconstruction")):
            axes[row, i].imshow(image.cpu(), origin="lower", aspect="auto", cmap="magma", vmin=0, vmax=1)
            axes[row, i].set_title(title)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "reconstructions.png", dpi=150)
    plt.close(fig)


def main() -> None:
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_segments = build_feature_cache(AUDIO_DIR, OUTPUT_DIR / "features")
    train_segments, validation_segments, test_segments = split_by_song(all_segments)
    print(f"segments: train={len(train_segments)}, validation={len(validation_segments)}, test={len(test_segments)}")

    train_loader = DataLoader(FeatureDataset(train_segments), batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(FeatureDataset(validation_segments), batch_size=BATCH_SIZE)
    test_loader = DataLoader(FeatureDataset(test_segments), batch_size=BATCH_SIZE)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicAutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_state, best_validation = None, float("inf")

    for epoch in range(1, EPOCHS + 1):
        train_mse = reconstruction_loss(model, train_loader, device, optimizer)
        validation_mse = reconstruction_loss(model, validation_loader, device)
        if validation_mse < best_validation:
            best_validation = validation_mse
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        if epoch == 1 or epoch % 5 == 0:
            print(f"Epoch {epoch:02d}/{EPOCHS}: train_MSE={train_mse:.6f}, val_MSE={validation_mse:.6f}")

    model.load_state_dict(best_state)
    test_mse = reconstruction_loss(model, test_loader, device)
    test_embeddings, test_metadata = embed_segments(model, test_segments, device)
    precision_at_5, mrr = retrieval_metrics(test_embeddings, test_metadata, k=5)
    print(f"Test reconstruction MSE: {test_mse:.6f}")
    print(f"Test retrieval Precision@5: {precision_at_5:.4f}, MRR: {mrr:.4f}")

    # 推薦時に使う全断片の埋め込みを保存する。
    all_embeddings, all_metadata = embed_segments(model, all_segments, device)
    np.save(OUTPUT_DIR / "embeddings.npy", all_embeddings)
    all_metadata.to_csv(OUTPUT_DIR / "metadata.csv", index=False)
    torch.save(model.state_dict(), OUTPUT_DIR / "autoencoder.pth")
    save_reconstruction_figure(model, test_loader, device)
    print("推薦例（embedding_id=0 を好きな断片とした場合）:")
    print(recommend(all_embeddings, all_metadata, liked_ids=[0]).to_string(index=False))


if __name__ == "__main__":
    main()
