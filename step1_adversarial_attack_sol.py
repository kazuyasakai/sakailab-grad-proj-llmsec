import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
from torchvision import models

# ART (Adversarial Robustness Toolbox)
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

def attack(epsilon, batch_size, output_file):
    """
    脆弱性分析のメインロジック
    __main__から渡されたパラメータに基づいて実行される
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 研究フェーズ開始 (Device: {device})")

    # 1. モデルのロード (最新の Weights 指定方式)
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.to(device)
    model.eval()

    # 2. ART Classifier の設定
    classifier = PyTorchClassifier(
        model=model,
        clip_values=(0.0, 1.0),
        loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224),
        nb_classes=1000,
        device_type=device.type
    )

    # 3. テストデータの生成
    x_test = np.random.rand(batch_size, 3, 224, 224).astype(np.float32)
    
    # 4. 攻撃実行 (FastGradientMethod)
    print(f"解析対象イプシロン: {epsilon}")
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_test_adv = attack.generate(x=x_test)

    # 5. 推論と評価
    preds_clean = np.argmax(classifier.predict(x_test), axis=1)
    preds_adv = np.argmax(classifier.predict(x_test_adv), axis=1)
    
    success_count = np.sum(preds_adv != preds_clean)
    success_rate = (success_count / batch_size) * 100
    print(f"攻撃成功率: {success_rate}% ({success_count}/{batch_size})")

    # 6. JSONログの保存
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "epsilon": epsilon,
        "success_rate": success_rate,
        "accuracy": 100.0 - success_rate,
        "clean_labels": preds_clean.tolist(),
        "adv_labels": preds_adv.tolist()
    }
    with open(output_file, 'w') as f:
        json.dump(log_data, f, indent=4)
    print(f"分析ログを保存しました: {output_file}")

    # 7. 可視化
    plt.figure(figsize=(15, 5))
    
    # Clean
    plt.subplot(1, 3, 1)
    plt.title(f"Clean (Label: {preds_clean})")
    plt.imshow(np.transpose(x_test[0], (1, 2, 0)))
    
    # Adversarial
    plt.subplot(1, 3, 2)
    plt.title(f"Adv (Label: {preds_adv})")
    plt.imshow(np.transpose(x_test_adv[0], (1, 2, 0)))
    
    # Perturbation (x10)
    plt.subplot(1, 3, 3)
    plt.title(f"Perturbation (diff x10)")
    diff = x_test_adv[0] - x_test[0]
    plt.imshow(np.clip(np.transpose(diff, (1, 2, 0)) * 10 + 0.5, 0, 1))
    
    plt.tight_layout()
    # 画像ファイルとしても保存
    plt.savefig(output_file.replace('.json', '.png'))
    plt.show()

if __name__ == "__main__":
    # --- エントリーポイントでのオプション処理 ---
    parser = argparse.ArgumentParser(description='AI Security Research - Phase 1')
    parser.add_argument('--eps', type=float, default=0.1, help='Epsilon (Attack Intensity)')
    parser.add_argument('--size', type=int, default=5, help='Batch Size')
    parser.add_argument('--output', type=str, default=None, help='Output filename')
    
    args = parser.parse_args()
    
    # 出力ファイル名の自動決定
    final_output = args.output if args.output else f"exp_eps_{args.eps}.json"
    
    # main() 関数へパラメータを渡して実行
    attack(epsilon=args.eps, batch_size=args.size, output_file=final_output)