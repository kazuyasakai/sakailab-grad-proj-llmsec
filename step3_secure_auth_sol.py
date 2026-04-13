import torch
import torch.nn as nn
import numpy as np
import argparse
import json
from datetime import datetime
from torchvision import models

# ART (Adversarial Robustness Toolbox)
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier

class SecureAILoginProtocol:
    """
    3ヶ月目の成果：検知とアクセス制御を統合した認証プロトコル
    """
    def __init__(self, classifier, threshold=0.02):
        self.classifier = classifier
        self.threshold = threshold
        self.auth_log = []

    def verify_request(self, input_data, user_id):
        # 1. 検知フェーズ（2ヶ月目のロジック）
        noise_sigma = 0.02
        x_recon = np.clip(input_data + np.random.normal(0, noise_sigma, input_data.shape), 0, 1)
        recon_error = np.mean(np.abs(input_data - x_recon), axis=(1, 2, 3))
        
        is_adversarial = recon_error > self.threshold
        
        results = []
        for i, attacked in enumerate(is_adversarial):
            status = "REJECTED" if attacked else "ACCEPTED"
            prediction = None
            
            # 2. 制御フェーズ：攻撃と判定された場合は推論をスキップ（フェイルセーフ）
            if not attacked:
                pred_raw = self.classifier.predict(input_data[i:i+1])
                prediction = int(np.argmax(pred_raw, axis=1)[0])
            
            # 3. ログ記録（監査証跡）
            auth_entry = {
                "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
                "status": status,
                "error_score": float(recon_error[i]),
                "ai_prediction": prediction
            }
            results.append(auth_entry)
            self.auth_log.append(auth_entry)
            
        return results

'''
再構成誤差のロジックをセンサーとして使い、画像に不自然な過敏性（攻撃の痕跡）がないかをチェック
正常なデータ: 誤差スコアが閾値 tau のとき、クリーンと判断
攻撃データ: 誤差スコアが閾値 tau 超のとき、異常（Adversarial）判断
'''
def secure_auth(epsilon, threshold, user_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- 3ヶ月目：認証プロトコル運用シミュレーション ---")

    # モデル・分類器準備
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights).to(device)
    model.eval()
    classifier = PyTorchClassifier(
        model=model, clip_values=(0.0, 1.0), loss=nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=0.01),
        input_shape=(3, 224, 224), nb_classes=1000, device_type=device.type
    )

    # プロトコルの初期化
    protocol = SecureAILoginProtocol(classifier, threshold=threshold)

    # テストデータの生成（1枚の画像から正常系と攻撃系を作成）
    x_clean = np.random.rand(1, 3, 224, 224).astype(np.float32)
    attack = FastGradientMethod(estimator=classifier, eps=epsilon)
    x_adv = attack.generate(x=x_clean)
    
    # シミュレーション実行
    print(f"\n[USER: {user_id}] 正常なアクセス試行中...")
    res_clean = protocol.verify_request(x_clean, user_id=user_id)[0]
    print(res_clean)
    print(f" -> 結果: {res_clean['status']}, 推論ID: {res_clean['ai_prediction']}")

    print(f"\n[USER: ATTACKER] 攻撃的なアクセス試行中 (eps={epsilon})...")
    res_adv = protocol.verify_request(x_adv, user_id="Unknown_Attacker")[0]
    print(f" -> 結果: {res_adv['status']}, 異常スコア: {res_adv['error_score']:.4f}")

    # 監査ログの保存
    log_file = "audit_log_final.json"
    with open(log_file, "w") as f:
        json.dump(protocol.auth_log, f, indent=4)
    print(f"\n監査ログを '{log_file}' に保存しました。")

# --- ここから開始されます ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase 3: Secure AI Authentication Protocol')
    parser.add_argument('--eps', type=float, default=0.1, help='Attack Epsilon')
    parser.add_argument('--threshold', type=float, default=0.02, help='Detection Threshold')
    parser.add_argument('--user', type=str, default="sakai", help='Simulated User ID')
    
    args = parser.parse_args()
    
    # 処理開始
    secure_auth(epsilon=args.eps, threshold=args.threshold, user_id=args.user)