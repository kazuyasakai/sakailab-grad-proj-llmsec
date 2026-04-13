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
        【TODO: 統合課題 1】
        1ヶ月目の攻撃、2ヶ月目の検知、3ヶ月目の判定ロジックをここに統合せよ。
        引数の epsilon と threshold に基づき、検知率(%)を計算して返すこと。
        """
        # ここに過去の成果を移植させる
        # 1. 攻撃画像生成
        # 2. 再構成誤差の計算
        # 3. 検知成功数のカウント
        return 0.0 # ダミー値

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
    # --- 【TODO: 統合課題 2】 ---
    # ネストされたループを用いて、全ての epsilon と threshold の組み合わせを評価せよ。
    # 結果は results_matrix (2次元リスト) に格納すること。
    
    # (ここを空欄にする)
    
    # --------------------------

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