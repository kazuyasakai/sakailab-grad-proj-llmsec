import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import models

# ART (Adversarial Robustness Toolbox) 関連のインポート
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

def main():
    # 1. デバイスの設定（GPUが使用可能な場合はGPUを使用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 学習済みモデル（ResNet18）のロード
    model = models.resnet18(pretrained=True)
    model.to(device)
    model.eval()  # 推論モードに設定

    # 3. 損失関数とオプティマイザの定義（ARTのClassifierに必要）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. ART PyTorchClassifier の作成
    # 入力値の範囲は通常 に正規化されることを想定
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=criterion,
        optimizer=optimizer,
        input_shape=(3, 224, 224),
        nb_classes=1000,
        device_type=device.type
    )

    # 5. 検証用データの準備（例としてランダムなノイズデータを使用）
    # 実際の実装では torchvision.datasets 等から正規の画像を使用すること（５枚の画像を生成）
    x_test = np.random.rand(5, 3, 224, 224).astype(np.float32)
    
    # 攻撃前の推論（Clean Predictions）
    preds_clean = classifier.predict(x_test)
    labels_clean = np.argmax(preds_clean, axis=1)
    print(f"Clean Predictions: {labels_clean}")

    # 6. FGSM 攻撃のインスタンス化と実行
    # eps (epsilon) は摂動の大きさ。この値を調整して脆弱性を分析する
    epsilon = 0.1
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_test_adv = attack.generate(x=x_test)

    # 7. 攻撃後の推論（Adversarial Predictions）
    preds_adv = classifier.predict(x_test_adv)
    labels_adv = np.argmax(preds_adv, axis=1)
    print(f"Adversarial Predictions: {labels_adv}")

    # 8. 攻撃成功率の算出
    # 元の判定と異なる判定になった割合を計算
    success_rate = np.sum(labels_adv != labels_clean) / len(labels_clean)
    print(f"Attack Success Rate (eps={epsilon}): {success_rate * 100}%")

    # 9. 可視化（バッチから最初の1枚 を取り出す）
    plt.figure(figsize=(12, 4))
    
    # --- 攻撃前 (Clean) ---
    plt.subplot(1, 3, 1)
    plt.title(f"Clean (ID: {labels_clean})")
    # x_testには５枚の画像が入っているので、x_test[0] と指定して1枚だけに絞る
    plt.imshow(np.transpose(x_test[0], (1, 2, 0)))
    
    # --- 攻撃後 (Adversarial) ---
    plt.subplot(1, 3, 2)
    plt.title(f"Adv (ID: {labels_adv})")
    # ここも：x_test_adv[0] と指定
    plt.imshow(np.transpose(x_test_adv[0], (1, 2, 0)))
    
    # --- 摂動 (Perturbation) の可視化 ---
    plt.subplot(1, 3, 3)
    plt.title("Perturbation (x10)")
    
    # x_test_adv[0] と x_test[0] を使い、1枚分の差分を計算する
    # 以前のコードが diff = x_test_adv - x_test になっていると、ここで4次元のままになります
    diff = x_test_adv[0] - x_test[0]
    
    # これで diff は (3, 224, 224) になり、transpose (1, 2, 0) が成功します
    plt.imshow(np.clip(np.transpose(diff, (1, 2, 0)) * 10 + 0.5, 0, 1))
    
    plt.show()

if __name__ == "__main__":
    main()