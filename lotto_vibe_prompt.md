# 로또(6/45) ‘이론 선택형 번호 생성/분석’ 시스템 설계 + 바이브 코딩용 역프롬프트

> ⚠️ 안내: 이 시스템은 **“당첨 예측”을 보장하지 않습니다.** 로또 추첨은 원칙적으로 독립·균등 무작위에 가깝고, 여기서 만드는 것은 **학습/엔터테인먼트 목적의 “번호 생성·필터링·백테스트” 도구**입니다.

---

## 1) 추천 아키텍처 (가장 구현 쉬운 조합)

### 현재 아키텍처 (FastAPI + Python + Tailwind)
- **Front**: Vanilla HTML + Tailwind CSS (Vibrant/Neon Design)  
- **Back/API**: FastAPI (Python) + Uvicorn
- **데이터 저장**: 로컬 JSON 기반 데이터베이스 (`data/` 폴더)
- **AI 모델**: PyTorch (Transformer, LSTM) 기반 예측 모듈
- **수집**: `scripts/update_data.py`를 통한 공식 당첨 데이터 동기화

**왜 추천?**
- UI 개발이 빠르고(컴포넌트 풍부), 서버/DB/인증이 단순해짐.
- “이론 모듈”을 플러그인처럼 추가하기 쉬움.
- 백테스트/통계 계산을 Edge Function/서버리스로 확장 가능.

### 옵션 B (Vue 선호 시)
- **Front**: Nuxt 3 + TypeScript + Tailwind + Nuxt UI(또는 Vuetify)
- **Back**: Supabase 동일
- 장점: Vue 생태계 친화 / 단점: shadcn 생태계(React)가 더 풍부

---

## 2) 핵심 개념: “이론을 선택해서 파이프라인을 구성”하는 구조

### 2.1. 파이프라인 개요
1) 데이터 로드 (회차별 당첨번호, 보너스, 날짜)  
2) **스코어링(가중치)**: 각 번호(1~45)에 점수 부여 (예: 핫/콜드, 누적 빈도, 미출현 등)  
3) **후보군 생성**: 상위 N개 번호 풀(pool) 구성 (예: 12~18개)  
4) **조합 생성**: pool에서 조합(6개) 생성 (전체 또는 샘플링)  
5) **필터링**: 홀짝/합/AC/구간 분포/끝수/델타 등 조건으로 제거  
6) **휠링**(선택): full/abbrev wheeling로 “구매할 게임 묶음” 구성  
7) **랭킹+설명**: 각 조합의 특징(홀짝, 합, AC, 끝수 분포 등)을 요약 + 왜 선택됐는지 설명  
8) **백테스트**: 과거 N회차에 대해 동일 파이프라인 적용, 성능(3/4/5/6개 일치 빈도, ROI 가정 등) 표시

### 2.2. “이론 모듈” 인터페이스(중요)
- `TheoryModule`
  - `id`, `name`, `type` = `"scorer" | "filter" | "generator" | "wheeling"`
  - `paramsSchema` (UI 폼 자동 생성용)
  - `run(input, params) -> output`
- 모듈은 JSON으로 등록되어 UI에서 체크/슬라이더/범위를 자동 렌더링

---

## 3) 데이터 수집(한국 로또 6/45 기준 예시)

- 공식 사이트에서 회차별 JSON을 받아오는 형태가 널리 쓰입니다. 예:
  - `https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=1000`
  - 응답은 JSON이며 `drwtNo1~6`, `bnusNo`, `drwNoDate`, `returnValue` 등을 포함합니다.

(구현에서는 DataProvider를 분리해 두고, 추후 다른 국가 로또도 추가 가능하게 만드세요.)

---

### 로컬 데이터 구조 (`data/`)
- `draws.json`: 회차별 당첨 번호, 보너스, 날짜 (로또 ID별 관리)
  - **한국**: 동행복권 (dhlottery.co.kr) JSON API 사용
  - **미국**: Data.gov / NC Lottery 공개 CSV/JSON 사용
- `generations.json`: AI 생성 이력, 생성 시간, 타겟 회차, 사용 모델 정보 포함
  - `transformer/`: Transformer 모델 정의, 가중치, 학습/생성 로직
  - `lstm/`: LSTM 모델 정의, 가중치, 학습/생성 로직
  - *참고: 미국 파워볼은 현재 당첨 번호만 수집하며, 상금 및 당첨자 정보는 미포함 (데이터 소스 제약)*
- `scripts/`:
  - `train.py`: 선택한 모델을 학습시키는 마스터 스크립트
  - `generate.py`: 선택한 모델로 번호를 생성하는 마스터 스크립트
  - `update_data.py`: 당첨 번호 동기화
  - `user_id uuid`
  - `name text`
  - `pipeline jsonb`  // 선택한 모듈 + 파라미터 전체
  - `created_at`
- `runs`
  - `id uuid pk`
  - `user_id uuid`
  - `config_id uuid`
  - `mode text` // generate/backtest
  - `result jsonb`
  - `created_at`
- `tickets`
  - `id uuid pk`
  - `run_id uuid`
  - `numbers int[]`  // [6]
  - `features jsonb` // sum, odd_even, ac, last_digits, deltas...
  - `score numeric`
  - `rank int`

### RLS
- `configs/runs/tickets`는 `user_id` 기반으로 Row Level Security.

---

## 5) API 설계 (서버리스 기준)

- `POST /api/draws/sync`
  - 입력: `{ from?: number, to?: number }`
  - 동작: 공식 JSON API를 호출해 draws upsert
- `POST /api/generate`
  - 입력: `{ configId, count, budget?, sampleMode? }`
  - 출력: 추천 조합 + 특징/설명
- `POST /api/backtest`
  - 입력: `{ configId, lastN }`
  - 출력: 적중 분포(0~6), 상위 1% 조합 특성, 재현성(랜덤 시드) 등
- `POST /api/wheel`
  - 입력: `{ pool: number[], wheelType: "full"|"abbrev", guarantee?: "4-if-6-in-10" ... }`
  - 출력: 게임 리스트

---

## 6) 기본 포함 “이론 모듈” 목록(당신이 제시한 이론을 모두 UI에서 선택 가능하게)

### Scorer(가중치)
- 누적 빈도/편차(LLN 기반)
- 핫/콜드 (최근 k회, 스코어 함수 선택: linear/exp)
- 미출현(스킵) 점수
- 회귀(예: N회귀: t-N 회차 번호 가중치)
- 끝수(0~9) 가중치

### Generator(생성)
- 가중치 기반 샘플링(Weighted sampling)
- 델타 기반 생성(Delta system): 델타 시퀀스 샘플링 → 누적합 → 정렬/중복 제거
- 시드 고정 모드(백테스트 재현성)

### Filter(필터)
- 홀짝 비율(예: 3:3, 4:2)
- 합계 범위(예: 100~175)
- 고저 비율(1~22 vs 23~45)
- 구간 전멸 규칙(10단위)
- AC값 범위(예: 7~10)  ※ 정의는 아래 참고
- 끝수 중복 제한(동형끝수 최대 2개 등)
- 델타 분포 제한(큰 간격 과다 방지)

### Wheeling
- Full wheel: nC6
- Abbrev wheel: 목표 보장 조건(템플릿) + 휴리스틱 최소화(그리디)

### AI 모델 (확장형 구조)
- **Transformer**: 시퀀스 위치 인코딩 기반의 고성능 예측 모델.
- **LSTM**: 시계열 특화 모델로, 과거 데이터의 패턴을 반영.
- **공통 기능**: 
    - Temperature 제어를 통한 생성 다양성 확보.
    - Top-K 필터링을 통해 확률 높은 번호 군집 추출.
    - 모델별 구분 라벨(Badge) 및 이력 관리 지원.

---

## 7) AC값 계산(참고 구현)
- 모든 쌍의 절대차 목록 → **중복 제거 후 고유 차이 개수 D**  
- `AC = D - (r-1)` (r=6이면 `AC = D - 5`)  
- 6/xx 로또에서 AC 범위는 보통 0~10

---

# 8) 바이브 코딩용 ‘역프롬프트’ (이대로 복붙해서 코딩 에이전트에 주면 됨)

## 역할
너는 시니어 풀스택 엔지니어다. “로또 이론 선택형 번호 생성/분석 도구”를 **학습/엔터테인먼트 목적**으로 만든다. 당첨 보장/조작/불법은 금지한다.

## 기술 스택 (현재 구현 기준)
- **Backend**: FastAPI (Python 3.10+)
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS (CDN)
- **AI Framework**: PyTorch
- **Storage**: Local JSON (FastAPI 런타임 내 절대 경로 처리)

## 핵심 요구사항
1) **이론 모듈을 체크박스로 선택**하고, 각 모듈 파라미터를 UI에서 설정할 수 있어야 한다.  
2) 선택 결과는 `pipeline json`으로 저장/불러오기 가능해야 한다.  
3) “Generate”는 추천 조합 N개를 만들고, 각 조합의 특징(홀짝/합/AC/구간/끝수/델타)을 함께 표시.  
4) “Backtest”는 최근 N회차에 대해 동일 파이프라인을 적용하고 적중 분포를 시각화.  
5) 데이터는 `draws` 테이블에 저장. 동기화 API가 draws를 upsert.  
6) UI는 “예측이 아니라 생성/분석 도구”임을 고정 안내문으로 표시.

## 구현 순서(반드시 이 순서로)
### Step 1. 프로젝트/의존성
- Next.js 앱 생성
- shadcn 설치
- Supabase 프로젝트 연결(환경변수: URL, ANON_KEY)

### Step 2. DB 스키마 + RLS
- `draws/configs/runs/tickets` 테이블 생성 SQL 제공
- RLS 정책: user 기반 접근

### Step 3. 데이터 동기화
- `/api/draws/sync` 구현: `drwNo` 범위 입력 받기
- 외부 JSON API 호출 후 `draws` upsert
- 실패/성공 로깅

### Step 4. 이론 모듈 프레임워크
- `/lib/theories/registry.ts`: 모듈 레지스트리(JSON + TS 타입)
- `scorer/filter/generator/wheeling` 공통 인터페이스
- 파이프라인 실행기 `/lib/engine/runPipeline.ts`

### Step 5. UI(핵심)
- 페이지: `/` (대시보드), `/config/[id]` (설정), `/run/[id]` (결과)
- 컴포넌트:
  - TheorySelector (체크 + 파라미터 폼)
  - GeneratePanel (count, budget, seed)
  - BacktestPanel (lastN)
  - ResultsTable (조합 + feature chips)
  - Charts (적중 분포 bar chart)

### Step 6. 알고리즘 최소 기능(먼저 단순하게)
- 기본 Generator: weighted sampling + 조합 생성
- Filter: odd/even + sum + high/low + AC
- Features 계산 유틸: sum, oddEven, highLow, lastDigitCounts, deltas, AC

### Step 7. 저장/불러오기
- configs 저장/불러오기
- runs/tickets 저장
- “export csv” 기능(옵션)

## 품질 기준(완료 조건)
- 새 config 생성 → 이론 선택/파라미터 설정 → generate 20개 → 결과가 DB에 저장되고 화면에 표시
- backtest last 50 실행 → 적중 분포가 그래프로 표시
- 코드에 “예측 보장 없음” 문구가 상단 고정
- 모든 API 입력 검증(zod) + 에러 메시지 UI 표시

## 코드 스타일
- 함수는 순수 함수 중심, 재현성 위해 random seed 지원
- 모듈은 확장 가능하게, theory 추가 시 registry에만 등록하면 UI에 자동 노출

---

## 9) (선택) 다음 확장 아이디어
- “예산”을 입력하면 휠링/조합 수를 자동 산정
- “대중 선호 조합 회피(생일 범위 1~31 과다 회피)” 같은 ‘상금 분할 최소화’ 실험 모드
- 모델 학습(ML)은 별도 마이크로서비스로 분리해 리소스 격리

