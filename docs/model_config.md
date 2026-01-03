# 🎰 모델 설정 및 ONNX 가이드
> 최종 업데이트: 2026-01-02

## 1. 지원 로또 및 모델 현황

| 로또 | ID | 공 개수 | 범위 | 모델 지원 현황 |
|------|----|---------|------|----------------|
| 🇰🇷 Korea 6/45 | `korea_645` | 6 | 1-45 | ✅ Transformer, LSTM, Vector |
| 🇺🇸 USA Powerball | `usa_powerball` | 5* | 1-69 | ✅ Transformer, LSTM (Main 5 only) |
| 🇺🇸 Mega Millions | `usa_megamillions` | 5* | 1-70 | ✅ Transformer, LSTM (Main 5 only) |
| 🇨🇦 Canada 6/49 | `canada_649` | 6 | 1-49 | ✅ Transformer, LSTM |
| 🇯🇵 Japan Loto 6 | `japan_loto6` | 6 | 1-43 | ⚠️ LSTM Only (TF 실패 → Korea 대용) |

*> 미국 로또의 경우 보너스 볼(Powerball/MegaBall)은 제외하고 메인 5개 번호만 예측합니다.*

---

## 2. ONNX 정적 모델 구조 (`web-static`)

웹 브라우저(Static Mode)에서 실행되는 모델들은 다음 폴더 구조로 관리됩니다.

```
web-static/models/
├── transformer/
│   ├── korea_645.onnx
│   ├── usa_powerball.onnx
│   └── ...
├── lstm/
│   ├── korea_645.onnx
│   └── ...
└── (vector, hot_trend 등은 JS 내부 로직으로 구현)
```

### 🔄 동적 로딩 및 Fallback 정책
1. **Dynamic Loading**: 사용자가 로또를 변경하면 `models/[type]/[lottery_id].onnx`를 로드합니다.
2. **Fallback**: 만약 전용 모델 파일이 없거나 로드에 실패하면(예: Japan TF), 자동으로 **`korea_645.onnx` (대표 모델)**을 로드하여 서비스 중단을 방지합니다.
3. **Adaptive Generation**: 불러온 모델이 무엇이든, 현재 선택된 로또의 설정(`ball_count` 등)에 맞춰 번호를 생성합니다. (예: 한국 모델로 미국 로또 생성 시 5개만 출력).

---

## 3. 학습 설정 가이드 (`config/`)

### 로또 설정 (`config/lotteries.json`)
```json
{
  "korea_645": {
    "name": "Korea Lotto 6/45",
    "ball_count": 6,
    "ball_range": [1, 45],
    "bonus": true
  }
}
```

### 학습 파라미터 (`config/training_config.json`)
| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `draw_length` | 10 | 입력으로 사용할 과거 회차 수 |
| `d_model` | 64 | 모델 크기 (Transformer Embed Dim) |
| `epochs` | 50 | 학습 반복 횟수 |

---

## 4. 앙상블 (Ensemble)
여러 모델의 예측 결과를 결합하여 신뢰도를 높입니다.

- **Weight**: Transformer(0.45) + LSTM(0.35) + Physics/Vector(0.20)
- **Logic**: 각 모델이 예측한 번호에 가중치 점수를 부여하고, 합산 점수가 높은 Top N개를 최종 추천.
- **Status**: ✅ 구현 완료 (Python & JS)

---

## 5. 향후 계획 (To-Do)
- [ ] **Bonus Ball 지원**: 미국 로또 등에서 보너스 볼까지 예측하도록 모델 출력 차원 확장.
- [ ] **Japan Loto 6 Transformer**: 학습 데이터 점검 및 모델 재학습으로 변환 에러 해결.
