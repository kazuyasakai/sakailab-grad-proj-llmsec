import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# ART (Adversarial Robustness Toolbox)
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

class SecurityEvaluator:
    def __init__(self, classifier):
        self.classifier = classifier

    def run_simulation(self, epsilon, threshold, n_samples=20):
        """
        特定のパラメータ下での防御性能を計測する（2・3ヶ月目のロジックを統合）
        """
        # 1. テストデータの準備
        x_clean = np.random.rand(n_samples, 3, 224, 224).astype(np.float32)
        attack = FastGradientMethod(estimator=self.classifier, eps=epsilon)
        x_adv = attack.generate(x=x_clean)

        # 2. 検知ロジック（再構成誤差ベース）
        noise_sigma = 0.02
        def calculate_error(data):
            # 画像をわざと汚して、その崩れ具合を測る
            recon = np.clip(data + np.random.normal(0, noise_sigma, data.shape), 0, 1)
            return np.mean(np.abs(data - recon), axis=(1, 2, 3))

        err_clean = calculate_error(x_clean)
        err_adv = calculate_error(x_adv)

        # 3. 統計量の算出
        tp = np.sum(err_adv > threshold)   # 攻撃を正しく検知
        fp = np.sum(err_clean > threshold) # 正常を誤検知
        
        # 検知率 (Recall) を計算
        detection_rate = (tp / n_samples) * 100
        return detection_rate

def main():
    print(f"--- 4ヶ月目：統合評価システム (Pandas不要版) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # モデルの準備
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device).eval()
    classifier = PyTorchClassifier(
        model=model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224), nb_classes=1000, device_type=device.type
    )

    evaluator = SecurityEvaluator(classifier)

    # 実験パラメータ（スイープ対象）
    epsilons = [0.0, 0.05, 0.1, 0.2, 0.3]
    thresholds = [0.012, 0.014, 0.016, 0.018, 0.020]
    
    # 結果格納用の2次元配列（純粋なPythonリスト）
    results_matrix = []

    print("シミュレーション実行中...")
    for eps in epsilons:
        row = []
        for tau in thresholds:
            # 攻撃強度が0の場合は「正常時」の検知（誤検知）テストになる
            det_rate = evaluator.run_simulation(eps, tau)
            row.append(det_rate)
            print(f"  [Eps: {eps}, Tau: {tau}] -> Det Rate: {det_rate}%")
        results_matrix.append(row)

    # --- 可視化フェーズ (Matplotlib のみ) ---
    results_array = np.array(results_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # imshow でヒートマップを描画
    im = ax.imshow(results_array, cmap="YlGnBu")

    # 軸の設定
    ax.set_xticks(np.arange(len(thresholds)))
    ax.set_yticks(np.arange(len(epsilons)))
    ax.set_xticklabels(thresholds)
    ax.set_yticklabels(epsilons)

    # 各セルに数値を表示
    for i in range(len(epsilons)):
        for j in range(len(thresholds)):
            text = ax.text(j, i, f"{results_array[i, j]:.1f}",
                           ha="center", va="center", color="black")

    ax.set_title("AI Security Performance Map (Detection Rate %)")
    ax.set_xlabel("Threshold (tau)")
    ax.set_ylabel("Attack Intensity (epsilon)")
    fig.colorbar(im, ax=ax, label='Detection Rate (%)')
    
    plt.tight_layout()
    plt.savefig("security_report_final.png")
    print("\n分析レポートを 'security_report_final.png' に保存しました。")
    plt.show()

if __name__ == "__main__":
    main()