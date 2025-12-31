# 로또 AI 분석 프로젝트

다국가 로또 데이터 수집 및 AI 분석 플랫폼.

## 프로젝트 구조

```
lotto/
├── api/
│   └── main.py             # FastAPI 서버
├── collectors/             # 데이터 수집 모듈
├── config/
│   └── lotteries.json      # 로또 설정
├── data/
│   └── korea_645/          # 로또 데이터 및 생성 이력
├── lotto_models/           # AI 모델 관리
│   ├── transformer/        # Transformer 모델 및 생성기
│   └── lstm/               # LSTM 모델 및 생성기
├── scripts/
│   ├── update_data.py      # 당첨 번호 업데이트
│   ├── train.py            # 모델 학습 (총괄)
│   └── generate.py         # 통합 번호 생성 (총괄)
├── web/
│   └── index.html          # 프론트엔드 UI
└── venv/                   # 가상환경
```

## 📊 데이터 출처 (Data Sources)
- **대한민국 로또 6/45**: [동행복권](https://www.dhlottery.co.kr/) API
- **미국 파워볼 (Powerball)**: [Data.gov (NY Lottery)](https://data.ny.gov/) 및 [NC Education Lottery](https://nclottery.com/) 공개 데이터 (※ 회차 편의를 위해 과거 데이터부터 역순(순차적)으로 일련번호를 부여하였습니다.)
- **캐나다 로또 6/49**: [GitHub (CorentinLeGuen/lotto-6-49-api)](https://github.com/CorentinLeGuen/lotto-6-49-api) SQLite 데이터 (1982-2023)
- **일본 ロト6**: [GitHub (tank1159jhs/jp-lottery-api)](https://github.com/tank1159jhs/jp-lottery-api) JSON 데이터 (최근 데이터만 제공)

## 설치

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 사용법

```bash
# 데이터 업데이트
python scripts/update_data.py --lottery korea_645

# 모델 학습 (CLI)
python scripts/train.py --model transformer --epochs 50
python scripts/train.py --model lstm --epochs 50

# 번호 생성 (CLI)
python scripts/generate.py --model transformer --count 5

# 스크립트로 간편하게 실행
./run.sh

# 또는 수동으로 실행
uvicorn api.main:app --reload
```

## 웹 UI 접속
서버 실행 후 브라우저에서 `http://localhost:8000` 접속

## 새 로또 추가

1. `config/lotteries.json`에 설정 추가
2. `collectors/`에 수집기 클래스 구현
3. `collectors/__init__.py`에 등록

## ⚠️ 면책

이 도구는 **학습/엔터테인먼트 목적**입니다. 당첨을 보장하지 않습니다.
