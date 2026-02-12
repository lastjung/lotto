#!/bin/bash

# 프로젝트 루트 디렉토리로 이동
cd "$(dirname "$0")"

# 가상환경 활성화 (venv 폴더가 있다고 가정)
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "⚠️ venv 폴더를 찾을 수 없습니다. pip install -r requirements.txt 를 먼저 실행해주세요."
    exit 1
fi

# API 서버 실행 (FastAPI + Uvicorn)
echo "🚀 AI 로또 분석기 서버를 시작합니다..."
echo "📍 접속 주소: http://localhost:8000"
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir ./api --reload-dir ./models_ai --reload-dir ./models_stat --reload-dir ./config
