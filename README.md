# 🎱 로또 AI 분석 프로젝트

다국가 로또 데이터 수집 및 AI 분석 플랫폼.

## 🌍 지원 로또

| 국가 | 로또 | 번호 범위 | 공 개수 | 모델 지원 |
|------|------|-----------|---------|-----------|
| 🇰🇷 한국 | 로또 6/45 | 1-45 | 6 | Transformer, LSTM, Vector, Physics Bias |
| 🇨🇦 캐나다 | 6/49 | 1-49 | 6 | Transformer, LSTM, Vector, Physics Bias |
| 🇯🇵 일본 | ロト6 | 1-43 | 6 | Transformer, LSTM, Vector, Physics Bias |
| 🇺🇸 미국 | Powerball | 1-69 | 5 | Transformer, LSTM, Physics Bias |
| 🇺🇸 미국 | Mega Millions | 1-70 | 5 | Transformer, LSTM, Physics Bias |

## 🤖 AI 모델

| 모델 | 설명 |
|------|------|
| **Transformer** | Attention-based 패턴 인식 |
| **LSTM** | Sequential Time-Series 분석 |
| **Vector** | 고차원 임베딩 + 클러스터링 |
| **Physics Bias** | 물리적 편향 (빈도/위치/트렌드) 분석 |

## 프로젝트 구조

```
lotto/
├── api/main.py                    # FastAPI 서버
├── config/
│   ├── lotteries.json             # 로또별 설정 (ball_range, ball_count)
│   └── training_config.json       # 학습 하이퍼파라미터
├── models_ai/src/
│   ├── transformer/               # Transformer 모델
│   ├── lstm/                      # LSTM 모델
│   └── vector/                    # Vector 모델
├── models_stat/
│   ├── physics_bias.py            # Physics Bias 모델
│   ├── ac_analysis.py             # AC 분석
│   └── sum_analysis.py            # 합계 분석
├── data/{lottery_id}/             # 로또별 데이터
└── web/index.html                 # 프론트엔드 UI
```

## 📊 데이터 출처

- **한국 로또 6/45**: [동행복권](https://www.dhlottery.co.kr/) API
- **미국 Powerball**: [Data.gov](https://data.ny.gov/) 공개 데이터
- **캐나다 6/49**: [GitHub (lotto-6-49-api)](https://github.com/CorentinLeGuen/lotto-6-49-api)
- **일본 ロト6**: [GitHub (jp-lottery-api)](https://github.com/tank1159jhs/jp-lottery-api)

## 설치 & 실행

```bash
# 가상환경 설정
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 서버 실행
uvicorn api.main:app --reload
```

## 모델 학습

```bash
# Transformer
python models_ai/src/transformer/train.py --lottery korea_645 --epochs 30
python models_ai/src/transformer/train.py --lottery usa_powerball --epochs 30

# LSTM
python models_ai/src/lstm/train.py --lottery korea_645 --epochs 30
```

## 웹 UI

서버 실행 후 `http://localhost:8000` 접속

## ⚠️ 면책

이 도구는 **학습/엔터테인먼트 목적**입니다. 당첨을 보장하지 않습니다.
