"""
PyTorch 모델을 ONNX로 변환하는 스크립트
Transformer 와 LSTM 모델을 브라우저에서 실행 가능한 ONNX 형식으로 변환
"""

import torch
import sys
from pathlib import Path

# 프로젝트 루트 추가
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models_ai.src.transformer.lotto_transformer import create_model as create_transformer
from models_ai.src.lstm.lotto_lstm import create_model as create_lstm


def convert_transformer_to_onnx(
    pt_path: str,
    onnx_path: str,
    seq_length: int = 10
):
    """Transformer 모델을 ONNX로 변환"""
    print(f"📦 Transformer 변환 시작: {pt_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    
    # 모델 생성 및 가중치 로드
    model = create_transformer(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 더미 입력 생성 (batch=1, seq_length=10, numbers=6)
    dummy_input = torch.randint(1, 46, (1, seq_length, 6))
    
    # ONNX export (legacy mode)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Legacy exporter 사용
    )
    
    print(f"✅ 변환 완료: {onnx_path}")
    return True


def convert_lstm_to_onnx(
    pt_path: str,
    onnx_path: str,
    seq_length: int = 10
):
    """LSTM 모델을 ONNX로 변환"""
    print(f"📦 LSTM 변환 시작: {pt_path}")
    
    # 체크포인트 로드
    checkpoint = torch.load(pt_path, map_location="cpu", weights_only=True)
    config = checkpoint.get("config", {})
    
    # 모델 생성 및 가중치 로드
    model = create_lstm(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 더미 입력 생성
    dummy_input = torch.randint(1, 46, (1, seq_length, 6))
    
    # ONNX export (legacy mode)
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        dynamo=False  # Legacy exporter 사용
    )
    
    print(f"✅ 변환 완료: {onnx_path}")
    return True


def main():
    """메인 변환 함수"""
    print("=" * 50)
    print("🔄 PyTorch → ONNX 변환 시작")
    print("=" * 50)
    
    # 경로 설정
    trained_dir = PROJECT_ROOT / "models_ai" / "trained"
    output_dir = PROJECT_ROOT / "web-static" / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    # Transformer 변환
    transformer_pt = trained_dir / "transformer" / "korea_645.pt"
    if transformer_pt.exists():
        try:
            convert_transformer_to_onnx(
                str(transformer_pt),
                str(output_dir / "transformer.onnx")
            )
            results.append(("Transformer", "✅ 성공"))
        except Exception as e:
            print(f"❌ Transformer 변환 실패: {e}")
            results.append(("Transformer", f"❌ 실패: {e}"))
    else:
        print(f"⚠️ Transformer 모델 없음: {transformer_pt}")
        results.append(("Transformer", "⚠️ 파일 없음"))
    
    # LSTM 변환
    lstm_pt = trained_dir / "lstm" / "korea_645.pt"
    if lstm_pt.exists():
        try:
            convert_lstm_to_onnx(
                str(lstm_pt),
                str(output_dir / "lstm.onnx")
            )
            results.append(("LSTM", "✅ 성공"))
        except Exception as e:
            print(f"❌ LSTM 변환 실패: {e}")
            results.append(("LSTM", f"❌ 실패: {e}"))
    else:
        print(f"⚠️ LSTM 모델 없음: {lstm_pt}")
        results.append(("LSTM", "⚠️ 파일 없음"))
    
    # 결과 출력
    print("\n" + "=" * 50)
    print("📊 변환 결과")
    print("=" * 50)
    for model, status in results:
        print(f"  {model}: {status}")
    
    # ONNX 파일 크기 확인
    print("\n📁 생성된 파일:")
    for onnx_file in output_dir.glob("*.onnx"):
        size_kb = onnx_file.stat().st_size / 1024
        print(f"  {onnx_file.name}: {size_kb:.1f} KB")


if __name__ == "__main__":
    main()
