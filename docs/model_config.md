# 🎰 모델 설정 가이드

> 작성일: 2026-01-01

## 변수명 규칙 (재학습 시 적용 예정)

### 모델 학습 Config (`models_ai/`)

| 현재 이름 | 새 이름 | 설명 | 예시 |
|-----------|---------|------|------|
| `num_numbers` | `ball_ranges` | 임베딩 크기 (max값) | 45, 49, 43 |
| `seq_length` | `history_length` | 입력 회차 수 | 10 |
| - | `ball_count` | 출력 공 개수 | 6, 5 |

### 로또 설정 Config (`config/lotteries.json`)

| 현재 이름 | 새 이름 | 설명 | 예시 |
|-----------|---------|------|------|
| `numbers_count` | `ball_count` | 뽑는 공 개수 | 6 |
| `number_range` | `ball_range` | 번호 범위 [min, max] | [1, 45] |

---

## 현재 파일 위치

```
📁 로또 설정
config/lotteries.json

📁 모델 코드 (기본값 정의)
models_ai/src/transformer/lotto_transformer.py
models_ai/src/lstm/lotto_lstm.py

📁 학습된 모델 (config 포함)
models_ai/trained/transformer/{lottery_id}.pt
models_ai/trained/lstm/{lottery_id}.pt
```

---

## 로또별 설정값

| 로또 | ball_range | ball_count | 모델 호환 |
|------|------------|------------|-----------|
| 🇰🇷 Korea 6/45 | [1, 45] | 6 | ✅ Transformer/LSTM |
| 🇨🇦 Canada 6/49 | [1, 49] | 6 | ⚠️ Vector 폴백 |
| 🇯🇵 Japan Loto6 | [1, 43] | 6 | ✅ Transformer/LSTM |
| 🇺🇸 Powerball | [1, 69] | 5 | ⚠️ Vector 폴백 |
| 🇺🇸 Mega Millions | [1, 70] | 5 | ⚠️ Vector 폴백 |

---

## 학습 설정 파일 (`config/training_config.json`)

### 모델 구조 파라미터

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `history_length` | 10 | 입력으로 사용할 과거 회차 수 |
| `d_model` | 64 | 임베딩 차원 (모델 크기) |
| `nhead` | 4 | 어텐션 헤드 수 (Transformer) |
| `num_layers` | 2 | Transformer/LSTM 레이어 수 |
| `dim_feedforward` | 128 | FFN 히든 크기 (Transformer) |
| `dropout` | 0.1 | 드롭아웃 비율 (과적합 방지) |

### 학습 파라미터

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `epochs` | 50 | 총 학습 반복 횟수 |
| `batch_size` | 32 | 한 번에 처리할 데이터 수 |
| `learning_rate` | 0.001 | 학습률 |
| `early_stopping` | 5 | N 에폭 개선 없으면 종료 |

### 자주 조정하는 값
- `history_length`: 더 긴 과거 참조 실험
- `epochs`: 학습 시간 조절
- `learning_rate`: 수렴 속도 조정

---

## TODO: 모델 재학습

Canada 649, Powerball 등에서 Transformer/LSTM 사용하려면:

1. 새 변수명으로 학습 스크립트 수정
2. 로또별 `ball_ranges` 값으로 개별 학습
3. `config/lotteries.json` 변수명 변경
4. `api/main.py` 참조 코드 업데이트

