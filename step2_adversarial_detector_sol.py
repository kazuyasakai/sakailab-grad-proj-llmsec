import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
from torchvision import models

# ART (Adversarial Robustness Toolbox) 関連
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

def detect_attack(epsilon, threshold):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    print(f"--- 2ヶ月目：検知ロジック実装フェーズ (Eps: {epsilon}, Threshold: {threshold}) ---")

    # 1. モデルと分類器の準備
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    
    classifier = PyTorchClassifier(
        model=model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224), nb_classes=1000, device_type=device.type
    )

    # 2. テストデータの生成（正常10枚 + 攻撃10枚）
    n_samples = 10
    x_clean = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
    
    # 攻撃データの生成 (1ヶ月目の手法)
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv = attack.generate(x=x_clean)

    # 評価用データセットの結合
    x_combined = np.concatenate([x_clean, x_adv])
    true_labels = np.concatenate([np.zeros(n_samples), np.ones(n_samples)]) # 0:正常, 1:攻撃

    # 3. 検知ロジックの実装（再構成誤差ベース）
    # 入力に微小なノイズを加えて「再構成」し、元の画像との差異（L1誤差）を計測する
    noise_sigma = 0.02
    x_reconstructed = np.clip(x_combined + np.random.normal(0, noise_sigma, x_combined.shape), 0, 1)
    
    # 各画像ごとの平均絶対誤差 (MAE) を計算
    reconstruction_error = np.mean(np.abs(x_combined - x_reconstructed), axis=(1, 2, 3))
    
    # 閾値判定
    detected_labels = (reconstruction_error > threshold).astype(int)

    # 4. 検知精度の評価
    tp = np.sum((true_labels == 1) & (detected_labels == 1)) # True Positive
    fp = np.sum((true_labels == 0) & (detected_labels == 1)) # False Positive
    fn = np.sum((true_labels == 1) & (detected_labels == 0)) # False Negative
    tn = np.sum((true_labels == 0) & (detected_labels == 0)) # True Negative
    
    recall = tp / n_samples
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"検知結果レポート:")
    print(f" - 攻撃検知数 (TP): {tp}/{n_samples}")
    print(f" - 正常誤検知 (FP): {fp}/{n_samples}")
    print(f" - 再現率 (Recall): {recall:.2f}")
    print(f" - 適合率 (Precision): {precision:.2f}")

    # 5. ファイル書き出し (JSON)
    output_data = {
        "config": {"epsilon": epsilon, "threshold": threshold, "noise_sigma": noise_sigma},
        "metrics": {"tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "recall": recall, "precision": precision},
        "raw_errors": reconstruction_error.tolist()
    }
    json_path = f"det_report_{timestamp}.json"
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"JSONログを保存しました: {json_path}")

    # 6. 可視化とグラフ保存
    plt.figure(figsize=(10, 6))
    plt.hist(reconstruction_error[:n_samples], bins=10, alpha=0.5, label='Clean', color='blue')
    plt.hist(reconstruction_error[n_samples:], bins=10, alpha=0.5, label='Adversarial', color='red')
    plt.axvline(threshold, color='green', linestyle='--', label=f'Threshold ({threshold})')
    plt.title(f"Reconstruction Error Distribution (eps={epsilon})")
    plt.xlabel("Mean Absolute Error")
    plt.ylabel("Frequency")
    plt.legend()
    
    png_path = f"det_dist_{timestamp}.png"
    plt.savefig(png_path)
    print(f"分布グラフを保存しました: {png_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 2: Adversarial Detector Evaluation')
    parser.add_argument('--eps', type=float, default=0.1, help='Attack intensity')
    parser.add_argument('--threshold', type=float, default=0.02, help='Detection threshold')
    args = parser.parse_args()
    
    detect_attack(args.eps, args.threshold)