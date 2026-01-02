# 🎱 로또 AI 시스템 업데이트 문서

> 최종 업데이트: 2026-01-01

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

---

## 8. 통계 분석 페이지 고도화 (2026-01-01)

### 8.1 기능 추가 (History Analysis)
- **Advanced Statistics**:
    - **Low/High Ratio**: 고저 비율 분석 (Low: 1~22, High: 23~45).
    - **End Digit Analysis**: 끝수(1의 자리) 출현 빈도 분석.
    - **AC Value Trend**: 산술적 복잡도(Arithmetic Complexity) 추세 그래프 (최근 30회).
    - **Bonus stats**: Hot/Cold 번호 카드에 보너스 번호 통계 추가.

### 8.2 UI/UX 개선
- **Dark Theme Re-skinning**:
    - 메인 사이트의 **"Deep Space"** 테마(Dark Purple) 적용.
    - Glassmorphism 카드 디자인 (`bg-[#1e293b]/50`, `backdrop-blur`).
    - Chart.js 다크 모드 가독성 최적화 (Grid/Font color adjustment).
- **Layout Optimization**:
    - 차트 그룹화: 비율(Ratio), 추세(Trend), 분포(Distribution) 로우(Row) 분리.
    - **Low/High 직관성 개선**: High(고)를 우측/상단에, Low(저)를 좌측/하단에 배치하여 시각적 혼동 제거.

---

## 9. web-vue (Quasar) 빈 화면 버그 수정 (2026-01-01)

### 9.1 문제 현상
- 앱 접속 시 **완전히 하얀 빈 화면**만 표시
- 브라우저 콘솔에 에러 없음 (Silent Failure)
- `#q-app` div는 존재하지만 내용이 비어있음
- Vue 앱이 마운트되지 않음

### 9.2 원인 분석
`quasar.config.js`의 `extendViteConf`에서 `.js`와 `.mjs` 파일을 `assetsInclude`에 포함:

```javascript
// ❌ 잘못된 설정
viteConf.assetsInclude = [...].concat(['**/*.onnx', '**/*.wasm', '**/*.mjs', '**/*.js'])
```

**결과**: Vite가 JavaScript 파일을 **정적 자산(static asset)**으로 취급하여 실행하지 않고 경로만 export:
```javascript
// 실제 코드 대신 이렇게 변환됨
export default "/.quasar/dev-spa/client-entry.js"
```

### 9.3 해결 방법
`.js`와 `.mjs`를 `assetsInclude`에서 제거:

```diff
# quasar.config.js (line 61)
- viteConf.assetsInclude = [...].concat(['**/*.onnx', '**/*.wasm', '**/*.mjs', '**/*.js'])
+ viteConf.assetsInclude = [...].concat(['**/*.onnx', '**/*.wasm'])
```

### 9.4 교훈
| 파일 확장자 | `assetsInclude`에 추가 가능 여부 |
|------------|--------------------------------|
| `.onnx` | ✅ 가능 (바이너리) |
| `.wasm` | ✅ 가능 (바이너리) |
| `.js` | ❌ 절대 불가 (실행 코드) |
| `.mjs` | ❌ 절대 불가 (ES 모듈) |
| `.ts` | ❌ 절대 불가 (실행 코드) |

> ⚠️ **주의**: `assetsInclude`는 이미지, 폰트, WASM 같은 **바이너리 파일**에만 사용해야 합니다.
> JavaScript/TypeScript 파일을 포함하면 Vite가 코드로 실행하지 않습니다.

### 9.5 추가 확인사항
- ONNX WASM 파일(`ort-wasm-simd-threaded.jsep.mjs`)이 404로 실패
- AI 모델 대신 통계 모델 폴백 사용 중
- 별도 수정 필요 (Phase 5 참조)

---

## 10. 용어 정리 (Draw vs History)

### 10.1 개념 구분

| 구분 | **Draw (추첨)** | **History (생성 기록)** |
|------|-----------------|------------------------|
| **의미** | 실제 로또 추첨 당첨 번호 | 사용자가 AI로 생성한 번호 기록 |
| **원천** | 동행복권, PowerBall 공식 데이터 | 앱에서 사용자가 생성 |
| **성격** | 불변 (과거 기록) | 누적 (사용자 활동) |
| **예시** | 1204회차: [3, 15, 22, 29, 35, 42] | Transformer 모델로 생성: [1, 5, 9, 18, 33, 40] |

---

### 10.2 JSON 파일 구조

#### Draw (추첨 데이터)
**파일**: `data/{lottery_id}/draws.json`
```json
{
  "draws": [
    {
      "draw_no": 1204,
      "draw_date": "2025-12-28",
      "numbers": [3, 15, 22, 29, 35, 42],
      "bonus": 17
    }
  ]
}
```

#### History (생성 기록) - LocalStorage
**키**: `lotto_history`
```json
[
  {
    "id": 1735689600000,
    "date": "2026-01-01T00:00:00Z",
    "model": "transformer",
    "lottery_type": "korea_645",
    "numbers": [1, 5, 9, 18, 33, 40],
    "analysis": { "sum": 106, "ac_value": 9 }
  }
]
```

---

### 10.3 Supabase 테이블 구조

| 테이블 | 용도 | 주요 컬럼 |
|--------|------|----------|
| `lotto_draws` | 추첨 당첨 번호 (공식 데이터) | `lottery_type`, `draw_number`, `numbers[]`, `bonus_number`, `draw_date` |
| `lotto_history` | 사용자 생성 기록 | `lottery_type`, `model`, `numbers (jsonb)`, `user_id`, `created_at` |

---

### 10.4 변수명 매핑

| 위치 | Draw 관련 | History 관련 |
|------|-----------|--------------|
| **Composable** | `useLotto.js` → `draws`, `loadDraws()` | `useHistory.js` → `history`, `saveEntry()` |
| **Props** | `LottoCharts` → `draws` | `HistoryList` → `history` |
| **Function Arg** | `generateWithAi(lottery, pastDraws)` | ✅ **수정 완료** |

---

### 10.5 변경 완료 (2026-01-01)

✅ **수정 완료**: `useAiEngine.js`의 `historyDraws`가 `pastDraws`로 변경되었습니다.

```javascript
// useAiEngine.js (line 66)
async function generateWithAi(lottery, pastDraws) {
    const recent = pastDraws.slice(0, windowSize)  // 최근 N회차 추출
}
```

**변경 이력**:
```diff
- async function generateWithAi(lottery, historyDraws)
+ async function generateWithAi(lottery, pastDraws)
```

---

### 10.6 통일 용어 가이드

| 영문 | 한글 | 사용처 |
|------|------|--------|
| `draw` / `draws` | 추첨, 추첨 번호 | 공식 당첨 데이터 |
| `history` / `entries` | 생성 기록 | 사용자 AI 생성 번호 |
| `draw_number` / `draw_no` | 회차 | 로또 추첨 회차 번호 |
| `generated_at` / `created_at` | 생성일 | 사용자가 번호를 생성한 시점 |

---

### 10.7 AI 모델 파라미터 용어

AI 모델(`models_ai/`)에서 사용되는 주요 파라미터들의 의미:

| 파라미터 | 의미 | 기본값 | 예시 |
|----------|------|--------|------|
| `ball_ranges` | 번호 범위 (1~N) | 45 | Korea 6/45 → 45, Powerball → 69 |
| `ball_count` | 추첨 공 개수 | 6 | Korea 6/45 → 6, Powerball → 5 |
| `draw_length` | **AI 입력 시퀀스 길이** | 10 | 이전 10회차 데이터로 다음 회차 예측 |

> ✅ **정리 완료**: `history_length`는 `draw_length`로 변경되어 "과거 추첨(Draw) 회차 수"임을 명확히 함.

#### 용어 변경 이력 (호환성)

```python
# lotto_transformer.py, lotto_lstm.py
# 이전 변수명 → 현재 변수명
num_numbers   → ball_ranges
seq_length    → draw_length
history_length→ draw_length  # 2026-01-01 추가
```

#### 모델 입출력 설명

```
입력:  (batch, draw_length, ball_count)
       예: (32, 10, 6) = 32개 배치, 이전 10회차, 각 회차 6개 번호

출력:  (batch, ball_count, ball_ranges)
       예: (32, 6, 45) = 32개 배치, 6개 번호 위치, 각 위치별 1~45번 확률
```

---

### 10.8 전체 용어 관계도

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA FLOW                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  [공식 데이터]           [AI 모델]              [사용자 기록]    │
│                                                                  │
│   draws.json  ─────────▶  Transformer  ─────────▶  lotto_history │
│   lotto_draws             LSTM                      (Supabase)   │
│   (Supabase)                                                     │
│                                                                  │
│   ┌─────────┐            ┌───────────┐           ┌────────────┐ │
│   │ draw_no │            │draw_      │           │ model      │ │
│   │ numbers │  (입력)    │length=10  │  (출력)   │ numbers    │ │
│   │ bonus   │ ────────▶  │ball_count │ ────────▶ │ lottery_   │ │
│   │ draw_   │            │ball_ranges│           │ type       │ │
│   │ date    │            └───────────┘           │ created_at │ │
│   └─────────┘                                    └────────────┘ │
│                                                                  │
│        Draw                                          History     │
│     (추첨 데이터)                                  (생성 기록)   │
└─────────────────────────────────────────────────────────────────┘
```


---

## 11. UI 복구 및 History 최적화 (2026-01-01) - 작성자: 재미나이 프래시 (Gemini Flash)

### 11.1 Tailwind CSS & PostCSS 환경 복구
- **문제**: PostCSS 설정 오류로 인해 Tailwind 유틸리티 클래스가 Vite에 의해 처리되지 않아 UI가 깨져 보이던 현상.
- **해결**: `postcss.config.js`를 ESM 기반 객체 설정 구문으로 전환하여 Vite/Quasar 빌드 시스템과의 호환성을 100% 복구함.

### 11.2 History 페이지 프리미엄 디자인 적용
- 대시보드의 **"Deep Space"** 테마를 History 페이지까지 확장 적용.
- **주요 개선 항목**:
    - 커스텀 탭 스위처 (글래스모피즘 및 네온 글로우 효과).
    - 히스토리 리스트 카드 디자인 고도화 및 호버 애니메이션 추가.
    - 요약 통계(Summary Stats) 섹션 신설.

### 11.3 데이터 정규화(Data Normalization) 시스템 구축
- **문제**: Supabase 및 LocalStorage의 데이터 포맷 불일치로 인해, 일부 데이터가 문자열이나 중첩 객체로 인식되어 `LottoBall` 렌더링이 비정상적으로 반복되던 버그(JSON 중복 출력 현상) 해결.
- **해결**: `useHistory.js`에 데이터 정규화 레이어를 구축하여, 원천 데이터의 형태(String, Object, Array)와 관계없이 항상 정제된 숫자 배열을 프론트엔드에 공급하도록 설계.

### 11.4 프로젝트 안정화 완료
- 모바일 앱 전환을 위한 UI 레이아웃 안정성 확보.

#### 시각적 증거 (Visual Proof)
![Dashboard UI](docs/images/dashboard_ui.png)
*그림 1: 복구된 딥 스페이스 테마 대시보드*

![History UI](docs/images/history_ui.png)
*그림 2: 데이터 정규화가 적용된 히스토리 리스트 (JSON 중첩 해결)*

---

## 12. 사이드바(왼쪽 날개) UI 전면 개편 (2026-01-01) - 작성자: 재미나이 프래시 (Gemini Flash)

### 12.1 브랜드 및 내비게이션 고도화
- **브랜드 업데이트**: 로고명을 **"LottoQuantAI"**로 변경하고, "Deep Space" 테마의 폰트 스타일을 적용하여 브랜드 정체성을 강화함.
- **아이콘 시스템 개선**: 기존의 로딩 문제(COEP)를 유발하던 외부 이미지를 제거하고, 신뢰성 높은 **Quasar/Material** 아이콘으로 교체하여 시스템 안정성을 확보함.

### 12.2 시각적 피드백 및 레이아웃 최적화
- **Active 상태 강화**: 현재 선택된 메뉴가 한눈에 들어오도록 **솔리드 블루 배경**과 화이트 폰트로 액티브 스타일을 보정함.
- **국가별 플래그**: 로또 선택기에 국가별 이모지 플래그를 추가하여 직관적인 필터링 경험을 제공함.

### 12.3 시스템 모니터링 패널 확장
- 시스템 상태 박스에 **"Agent: Connected"** 정보를 추가하여 AI 모델과의 실시간 연결 상태를 사용자에게 명확히 전달하도록 개선함.

#### 시각적 증거 (Visual Proof)
![Sidebar UI](docs/images/sidebar_ui.png)
*그림 3: 최종 사이드바 디자인 (LottoQuantAI 브랜드 및 아이콘 복구)*

#### ⚠️ UI 디스크립판시 알림 (Baseline Comparison)
![Legacy Sidebar](docs/images/legacy_sidebar_8000.png)
*그림 4: 기존 포트(8000)의 사이드바 디자인 - **현재 신규 버전(9001)과 날개 부분이 완전히 다름***

---

---
