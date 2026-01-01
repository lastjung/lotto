# 🎱 로또 AI 시스템 업데이트 문서

> 최종 업데이트: 2025-12-31

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

### 1.5 AC값 (Arithmetic Complexity) 분석

**파일**: `models_stat/ac_analysis.py`

AC값은 로또 번호 조합의 산술적 무작위성을 정량화한 지표입니다.

**알고리즘**:
1. 6개 번호에서 모든 쌍의 차이 계산 (C(6,2) = 15개)
2. 중복 제거 후 고유 차이 개수 U 계산
3. AC = U - (N - 1) = U - 5

**예시**:
| 번호 조합 | AC값 | 등급 |
|----------|------|------|
| 1, 2, 3, 4, 5, 6 | 0 | 매우 낮음 (피할 것) |
| 5, 10, 15, 20, 25, 30 | 0 | 매우 낮음 (피할 것) |
| 3, 15, 22, 29, 35, 42 | 9 | 높음 (추천) |

**시스템 적용**:
- AC < 7: 규칙적인 조합으로 판단 → **자동 폐기**
- AC ≥ 7: 무작위적인 조합 → 허용

---

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
├── models_ai/               # AI 딥러닝 모델 (Transformer, LSTM)
│   ├── src/                 # 모델 소스코드
│   └── trained/             # 학습된 모델 파일 (.onnx, .pt)
├── models_stat/             # 통계 분석 모듈
│   ├── ac_analysis.py       # AC값 분석
│   ├── sum_analysis.py      # 합계 기반 분석
│   └── consecutive_analysis.py # 연속 번호 분석
├── web/
│   ├── index.html           # 메인 프론트엔드 (V2)
│   └── js/                  # 프론트엔드 로직 (app.js, ui.js)
├── web-static/              # ONNX 전용 정적 배포본
└── .gitignore               # Git 제외 파일
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
| 🇰🇷 한국 | 6/45 | 1,204 | 2002년~ | ✅ 있음 | 동행복권 API |
| 🇺🇸 미국 | Powerball | 1,882 | 2010년~ | ❌ 없음 | NY Data.gov |
| 🇺🇸 미국 | Mega Millions | 2,462 | 2002년~ | ❌ 없음 | NY Data.gov |
| 🇨🇦 캐나다 | 6/49 | 4,152 | 1982~2025 | ❌ 없음 | GitHub + WCLC |
| 🇯🇵 일본 | ロト6 | 100 | 2025년 (최근) | ❌ 없음 | GitHub API |

### 3.2 모델 저장 구조 (5개 로또 × 2개 모델 = 10개)

```
models_ai/trained/
├── transformer/
│   ├── {lottery_id}.onnx    # ONNX (웹 브라우저용)
│   └── {lottery_id}.pt      # PyTorch (서버용)
└── lstm/
    └── {lottery_id}.onnx    # ONNX (웹 브라우저용)
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
| 캐나다 데이터 공백 | ⚠️ | 2024~2025년 데이터 수동 복구 완료 (스크래핑 연동 강화 필요) |
| 일본 과거 전체 데이터 | ❌ | GitHub에 최근 데이터만 존재 |

---

## 4. 향후 계획

- [ ] 미국/캐나다 상금 정보 확보 (유료 API 또는 스크래핑)
- [ ] 일본 로또 과거 데이터 수집 (1회~1964회)
- [ ] Vue/Quasar 프론트엔드 마이그레이션
- [ ] Supabase/Vercel 클라우드 배포

---

## 5. 배포 아키텍처 고려사항

### 5.1 Vercel + Supabase만 사용 시 문제점

| 문제 | 설명 |
|------|------|
| **PyTorch 실행 불가** | Vercel Serverless는 50MB 제한, PyTorch는 700MB |
| **Python 서버 없음** | AI 모델 추론할 곳이 없음 |

> ⚠️ **핵심 이슈**: Supabase는 데이터베이스이고, Vercel은 정적 호스팅.
> 둘 다 **PyTorch 코드를 실행할 수 없음**.

---

### 5.2 해결책 비교

| 해결책 | 방법 | 장점 | 단점 |
|--------|------|------|------|
| **ONNX** | 모델을 브라우저에서 실행 | 서버 비용 $0, 오프라인 가능 | 첫 로딩 느림, 저사양 기기 문제 |
| **Railway** | Python 서버 추가 | 빠른 추론, 모델 업데이트 쉬움 | 월 $5 비용 |

---

### 5.3 배포 옵션

#### 옵션 A: ONNX 방식 (무료)
```
Vercel (프론트엔드) + Supabase (DB)
├── ONNX 모델 파일 (.onnx) - 브라우저에서 로드
├── ONNX Runtime Web (JS) - 브라우저에서 추론
└── JSON 데이터 → Supabase로 마이그레이션
```

**특징**:
- 서버 비용 $0
- 브라우저에서 AI 추론
- 첫 로딩 시 모델 파일 다운로드 필요 (1~2MB)

#### 옵션 B: Railway 방식 (월 $5)
```
Vercel (프론트엔드) + Supabase (DB) + Railway (Python API)
├── 현재 FastAPI 코드 그대로 사용
├── PyTorch 모델 그대로 사용
└── ONNX 변환 필요 없음
```

**특징**:
- 현재 코드 변경 최소화
- 빠른 추론 속도
- 월 $5 크레딧 (무료 티어 내 가능)

---

### 5.4 예상 비용

| 서비스 | 무료 티어 | 유료 시 |
|--------|----------|---------|
| Vercel | ✅ 무료 | - |
| Supabase | ✅ 무료 (500MB) | $25/월 |
| Railway | ⚠️ 월 $5 크레딧 | $5~/월 |

---

### 5.5 ONNX 변환 완료 현황 (2025-12-31)

| 모델 | 원본 크기 | ONNX 크기 | 상태 |
|------|----------|----------|------|
| Transformer | 312KB (.pt) | 371KB (.onnx) | ✅ 완료 |
| LSTM | ~300KB (.pt) | 1.7MB (.onnx) | ✅ 완료 |
| Vector | - | JS 구현 | ✅ 완료 |

**파일 위치**: `web-static/models/`

---

### 5.6 web-static 데이터 관리

**로컬 개발**: 심볼릭 링크로 메인 데이터 참조
```
web-static/data/
└── korea_645.json -> ../../data/korea_645/draws.json
```

**배포 시**: 심볼릭 링크를 실제 파일로 복사
```bash
# -L 옵션: 심볼릭 링크를 따라가서 실제 파일 복사
cp -L web-static/data/*.json deploy/data/
```

| 환경 | 방식 |
|------|------|
| 로컬 개발 | 심볼릭 링크 (자동 동기화) |
| GitHub Pages/Vercel | 실제 파일 복사 필요 |

---

## 6. 시스템 현대화 및 자동화 (2025-12-31)

### 6.1 UI 코드베이스 통합 (Source of Truth)
- **통합 위치**: `/web` 디렉토리를 통합 소스 저장소로 설정.
- **동적 모드 탐지**: `app.js`에서 접속 포트(8000 vs 8081)를 감지하여 API 서버 모드와 ONNX 로컬 실행 모드를 자동 전환.
- **동기화 구조**: `web-static`은 심볼릭 링크를 통해 `web`의 코드를 참조하여 코드 중복을 제거하고 UI 일관성을 확보.

### 6.2 데이터 수집 시스템 표준화
- **BaseLotteryCollector**: 모든 수집기의 인터페이스를 표준화하여 상속 구조 개선.
- **Draw Dataclass**: 회차 데이터 형식을 데이터 클래스로 공식화하여 타입 안정성 강화.
- **증분 업데이트(Incremental Fill)**: 누락된 회차만 선별적으로 채우는 로직을 베이스 클래스에 내장.

### 6.4 프론트엔드 하이브리드 설정 UI (System Tab)
- **Dashboard**: 가장 빈번하게 사용되는 **Generation Count**만 메인 대시보드(Start 버튼 바로 위)에 배치하여 접근성 극대화.
- **System Tab (Sidebar)**: 전문적인 분석 필터들(AC, Sum, Consecutive)을 사이드바의 `System` 탭으로 분리 통합.
- **이점**:
    - **Clean UI**: 대시보드의 복잡도를 낮추고 분석 결과물에 집중할 수 있는 공간 확보.
    - **전문가 모드**: 상세한 통계 필터링을 원하는 사용자만 `System` 창에서 정밀하게 환경 설정 가능.
    - **확장성**: 추후 AI 가중치나 추가적인 수치 제한 옵션을 System 탭에서 쉽게 관리 가능.

### 6.5 프로젝트 상태 요약 (Done)
- [x] **UI Renovation**: Premium Glassmorphism V2 테마 적용 및 레이아웃 최적화.
- [x] **Filter Integration**: AC, 합계, 연속 번호 등 이론 필터들을 엔진에 통합.
- [x] **Sidebar Logic**: 대시보드와 설정(System) 영역의 논리적 분리 및 상태 유지 로직 구현.
- [x] **Data Completion**: 캐나다 로또 등 누락된 과거 데이터 수집 및 자동화 프로세스 구축.

---

## 7. 프론트엔드 정리 (2026-01-01)

### 7.1 ui.js / app.js 충돌 해결
| 문제 | 해결 |
|------|------|
| `displayResults` 오버라이드 | 제거 (-80줄) |
| `loadHistory` 오버라이드 | 제거 (-35줄) |
| `selectModel` 오버라이드 | 제거 (-60줄) |
| `supabase` 변수 충돌 | `supabaseClient`로 변경 |

### 7.2 결과 헤더 표시 개선
- **Before**: 번호만 표시
- **After**: `Korea Lotto 6/45 | HOT_TREND | Draw 1205` 헤더 추가

### 7.3 LSTM/Vector 자동 생성 수정
```diff
- if (!isInit && (type === 'transformer' || type === 'hot_trend')) {
+ if (!isInit && ['transformer', 'lstm', 'vector', 'hot_trend'].includes(type)) {
```

### 7.4 모든 모델 정상 작동 확인
| 모델 | 생성 | 저장 |
|------|------|------|
| Transformer | ✅ | ✅ |
| LSTM | ✅ | ✅ |
| Vector | ✅ | ✅ |
| Hot Trend | ✅ | ✅ |
