"""
PyTorch 모델을 ONNX로 변환하는 스크립트 (Multi-Lottery + Folder Structure)
모든 로또 종류에 대해 Transformer와 LSTM 모델을 ONNX 형식으로 변환하여
모델별 하위 폴더에 저장합니다.
"""

import torch
import sys
import json
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 모델 생성 함수 임포트
from models_ai.src.transformer.lotto_transformer import create_model as create_transformer
from models_ai.src.lstm.lotto_lstm import create_model as create_lstm


def convert_model_to_onnx(
    model_type: str,
    pt_path: Path,
    onnx_path: Path
):
    """모델을 ONNX로 변환하는 공통 함수 (config에서 설정 자동 로드)"""
    print(f"📦 {model_type.upper()} 변환 시작: {pt_path.name}")
    
    try:
        # 체크포인트 로드
        checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        
        # 설정에서 값 읽기 (기본값 포함)
        ball_count = config.get("ball_count", 6)
        ball_ranges = config.get("ball_ranges", config.get("num_numbers", 45))  # max number
        draw_length = config.get("draw_length", config.get("seq_length", 10))  # sequence length
        
        # 모델 생성
        if model_type == "transformer":
            model = create_transformer(config)
        elif model_type == "lstm":
            model = create_lstm(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        
        # 더미 입력 생성 (batch=1, draw_length, ball_count) - config 기반 동적 생성
        # 범위: 1 ~ ball_ranges (모델이 학습된 범위)
        dummy_input = torch.randint(1, ball_ranges + 1, (1, draw_length, ball_count))
        
        # 출력 폴더 생성
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        # ONNX export
        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            dynamo=False
        )
        print(f"  ✅ 생성 완료 (Balls: {ball_count}, Range: 1-{ball_ranges}, Seq: {draw_length}): {onnx_path.relative_to(PROJECT_ROOT)}")
        return True, ball_count
        
    except Exception as e:
        print(f"  ❌ 변환 실패: {e}")
        return False, 0


def main():
    print("=" * 60)
    print("🔄 Multi-Lottery ONNX 변환 (폴더 구조화)")
    print("=" * 60)
    
    trained_dir = PROJECT_ROOT / "models_ai" / "trained"
    output_dir = PROJECT_ROOT / "web-static" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 로또 설정 로드
    config_path = PROJECT_ROOT / "config" / "lotteries.json"
    if not config_path.exists():
        print("❌ config/lotteries.json 파일을 찾을 수 없습니다.")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        lotteries = json.load(f)
        
    print(f"📋 총 {len(lotteries)}개의 로또 설정 발견")
    
    results = []
    
    # 2. 각 로또별로 순회
    for lottery_id in lotteries.keys():
        print(f"\nTarget: {lottery_id}")
        
        # Transformer -> web-static/models/transformer/[id].onnx
        tf_pt = trained_dir / "transformer" / f"{lottery_id}.pt"
        if tf_pt.exists():
            tf_onnx = output_dir / "transformer" / f"{lottery_id}.onnx"
            success, bc = convert_model_to_onnx("transformer", tf_pt, tf_onnx)
            results.append((f"{lottery_id}/TF", success))
        else:
            print(f"  ⚠️  Transformer 모델 없음 ({tf_pt.name})")
            results.append((f"{lottery_id}/TF", "Skip"))

        # LSTM -> web-static/models/lstm/[id].onnx
        lstm_pt = trained_dir / "lstm" / f"{lottery_id}.pt"
        if lstm_pt.exists():
            lstm_onnx = output_dir / "lstm" / f"{lottery_id}.onnx"
            success, bc = convert_model_to_onnx("lstm", lstm_pt, lstm_onnx)
            results.append((f"{lottery_id}/LSTM", success))
        else:
            print(f"  ⚠️  LSTM 모델 없음 ({lstm_pt.name})")
            results.append((f"{lottery_id}/LSTM", "Skip"))

    # 3. 결과 요약
    print("\n" + "=" * 60)
    print("📊 전체 변환 결과")
    print("=" * 60)
    for name, status in results:
        status_icon = "✅" if status is True else "❌" if status is False else "⚠️"
        print(f"{status_icon} {name}: {status if isinstance(status, str) else 'OK'}")
        
    print(f"\n📂 저장 경로: {output_dir}")

if __name__ == "__main__":
    main()
