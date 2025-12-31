# 🎱 로또 AI 시스템 업데이트 문서

> 최종 업데이트: 2025-12-30

---

## 1. AI 모델 구축

### 1.1 사용 라이브러리

| 라이브러리 | 버전 | 용도 |
|-----------|------|------|
| **PyTorch** | 2.x | 딥러닝 프레임워크 |
| **FastAPI** | 0.x | REST API 서버 |
| **Uvicorn** | 0.x | ASGI 웹서버 |

### 1.2 Transformer 모델

**파일**: `lotto_models/transformer/lotto_transformer.py`

```python
class LottoTransformer(nn.Module):
    """로또 번호 예측용 소형 Transformer"""
    
    def __init__(
        self,
        num_numbers: int = 45,       # 로또 번호 범위 (1~45)
        seq_length: int = 10,        # 입력 시퀀스 길이 (이전 N회차)
        d_model: int = 64,           # 임베딩 차원
        nhead: int = 4,              # 어텐션 헤드 수
        num_layers: int = 2,         # Transformer 레이어 수
        dim_feedforward: int = 128,  # FFN 차원
        dropout: float = 0.1
    )
```

**핵심 컴포넌트**:
- `nn.Embedding`: 번호 → 벡터 변환
- `nn.TransformerEncoderLayer`: Self-Attention 레이어
- `PositionalEncoding`: 시퀀스 위치 정보 인코딩

**메서드**:
| 메서드 | 입력 | 출력 | 설명 |
|--------|------|------|------|
| `forward(x)` | (batch, seq, 6) | (batch, 6, 45) | 각 위치별 번호 확률 |
| `predict(x, temperature, top_k)` | 시퀀스 | (batch, 6) | 샘플링으로 번호 생성 |

---

### 1.3 LSTM 모델

**파일**: `lotto_models/lstm/lotto_lstm.py`

```python
class LottoLSTM(nn.Module):
    """로또 번호 예측용 LSTM 모델"""
    
    def __init__(
        self,
        num_numbers: int = 45,       # 로또 번호 범위
        seq_length: int = 10,        # 입력 시퀀스 길이
        embedding_dim: int = 64,     # 임베딩 차원
        hidden_dim: int = 128,       # LSTM hidden 차원
        num_layers: int = 2,         # LSTM 레이어 수
        dropout: float = 0.2
    )
```

**핵심 컴포넌트**:
- `nn.Embedding`: 번호 임베딩
- `nn.LSTM`: 순환 신경망 (시계열 패턴 학습)
- `nn.Linear`: 출력 레이어

---

### 1.4 학습 설정

```python
# 공통 학습 파라미터
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# 데이터셋 분할
train_size = 80%
val_size = 20%
batch_size = 32
epochs = 50
```

---

## 2. 시스템 구성

### 2.1 프로젝트 구조

```
lotto/
├── api/
│   └── main.py              # FastAPI 서버
├── collectors/
│   ├── base.py              # 베이스 수집기
│   ├── korea_645.py         # 🇰🇷 한국 로또
│   ├── usa_powerball.py     # 🇺🇸 미국 파워볼
│   ├── usa_megamillions.py  # 🇺🇸 미국 메가밀리언즈
│   ├── canada_649.py        # 🇨🇦 캐나다 6/49
│   └── japan_loto6.py       # 🇯🇵 일본 로또6
├── config/
│   └── lotteries.json       # 로또 설정
├── data/
│   ├── korea_645/           # 한국 데이터
│   ├── usa_powerball/       # 파워볼 데이터
│   ├── usa_megamillions/    # 메가밀리언즈 데이터
│   ├── canada_649/          # 캐나다 데이터
│   └── japan_loto6/         # 일본 데이터
├── lotto_models/
│   ├── src/                 # 모델 소스코드
│   │   ├── transformer/
│   │   │   ├── lotto_transformer.py
│   │   │   └── train.py
│   │   └── lstm/
│   │       ├── lotto_lstm.py
│   │       └── train.py
│   └── trained/             # 학습된 모델 파일 (.pt)
│       ├── transformer/
│       │   ├── korea_645.pt
│       │   ├── canada_649.pt
│       │   └── japan_loto6.pt
│       └── lstm/
│           ├── korea_645.pt
│           ├── canada_649.pt
│           └── japan_loto6.pt
├── scripts/
│   ├── train.py             # 학습 스크립트
│   └── update_data.py       # 데이터 업데이트
├── web/
│   └── index.html           # 프론트엔드 UI
├── .gitignore               # Git 제외 파일
└── run.sh                   # 서버 실행 스크립트
```

### 2.2 API 엔드포인트

| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/api/generate` | POST | AI 번호 생성 |
| `/api/history` | GET | 생성 이력 조회 |
| `/api/draws/{lottery_id}` | GET | 당첨 데이터 조회 |
| `/api/compare` | POST | 당첨 번호 비교 |

### 2.3 실행 방법

```bash
# 서버 실행
./run.sh

# 또는 수동 실행
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 3. 데이터 현황

### 3.1 국가별 데이터 (5개 로또)

| 국가 | 로또 | 수집 회차 | 데이터 기간 | 상금 정보 | 데이터 소스 |
|------|------|----------|------------|----------|------------|
| 🇰🇷 한국 | 6/45 | ~1,158 | 2002년~ | ✅ 있음 | 동행복권 API |
| 🇺🇸 미국 | Powerball | 1,882 | 2010년~ | ❌ 없음 | NY Data.gov |
| 🇺🇸 미국 | Mega Millions | 2,462 | 2002년~ | ❌ 없음 | NY Data.gov |
| 🇨🇦 캐나다 | 6/49 | 4,144 | 1982~2023 | ❌ 없음 | GitHub SQLite |
| 🇯🇵 일본 | ロト6 | 100 | 2025년 (최근) | ❌ 없음 | GitHub API |

### 3.2 모델 저장 구조 (5개 로또 × 2개 모델 = 10개)

```
lotto_models/
├── transformer/
│   ├── korea_645.pt        # 한국 전용 Transformer
│   ├── usa_powerball.pt    # 파워볼 전용
│   ├── usa_megamillions.pt # 메가밀리언즈 전용
│   ├── canada_649.pt       # 캐나다 전용
│   └── japan_loto6.pt      # 일본 전용
└── lstm/
    └── {lottery_id}.pt     # 동일 구조 (5개)
```

**학습 명령어**:
```bash
python scripts/train.py --model transformer --lottery korea_645
python scripts/train.py --model lstm --lottery usa_powerball
python scripts/train.py --model transformer --lottery all  # 전체 학습
```

### 3.2 데이터 구조

```json
{
  "draws": [
    {
      "draw_no": 1158,
      "draw_date": "2025-12-28",
      "numbers": [3, 15, 22, 29, 35, 42],
      "bonus": 17,
      "first_prize_amount": 2500000000,
      "first_prize_winners": 3
    }
  ],
  "updated_at": "2025-12-30T00:00:00",
  "lottery_id": "korea_645",
  "total_draws": 1158
}
```

### 3.3 제한 사항

| 항목 | 상태 | 설명 |
|------|------|------|
| 미국 파워볼 상금 정보 | ❌ | 공개 API에서 미제공 |
| 캐나다 2023년 이후 데이터 | ❌ | GitHub 소스 한계 |
| 일본 과거 전체 데이터 | ❌ | GitHub에 최근 데이터만 존재 |

---

## 4. 향후 계획

- [ ] 미국/캐나다 상금 정보 확보 (유료 API 또는 스크래핑)
- [ ] 일본 로또 과거 데이터 수집 (1회~1964회)
- [ ] Vue/Quasar 프론트엔드 마이그레이션
- [ ] Supabase/Vercel 클라우드 배포
