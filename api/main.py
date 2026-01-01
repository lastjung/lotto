"""
로또 AI 분석 FastAPI 서버
"""

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import json
from pathlib import Path
import sys
import random
from collections import Counter

# 프로젝트 루트를 sys.path에 추가 (앱 시작 시 한 번만)
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 모델 관련 임포트
from models_ai.src.transformer.lotto_transformer import create_model as create_transformer
from models_ai.src.lstm.lotto_lstm import create_model as create_lstm
from models_ai.src.vector.lotto_vector import LottoVectorModel
from models_stat.ac_analysis import calculate_ac, is_valid_ac, get_ac_rating
from models_stat.sum_analysis import is_valid_sum
from models_stat.consecutive_analysis import has_consecutive
import torch

app = FastAPI(title="로또 AI 분석 API", version="1.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 데이터 경로 (절대 경로로 전환하여 Uvicorn 실행 환경 이슈 해결)
DATA_DIR = PROJECT_ROOT / "data"

# 모델 캐시 (메모리 절약 및 빠른 로딩용)
MODELS = {}

def get_model(model_type: str = "transformer", lottery_id: str = "korea_645"):
    """모델 로드 (lazy loading) - 로또별 개별 모델 지원"""
    global MODELS
    cache_key = f"{model_type}_{lottery_id}"
    
    if cache_key not in MODELS:
        # Vector 모델은 실시간 학습 (학습 파일 불필요)
        if model_type == "vector":
            draws = load_draws(lottery_id)
            model = LottoVectorModel(n_clusters=5, n_neighbors=10)
            model.fit([d["numbers"] for d in draws])  # dict에서 numbers 추출
            MODELS[cache_key] = model
            return MODELS[cache_key]
        
        folder = "transformer" if model_type == "transformer" else "lstm"
        
        # 1. 로또별 모델 파일 찾기 (우선)
        model_path = PROJECT_ROOT / "models_ai" / "trained" / folder / f"{lottery_id}.pt"
        
        # 2. 없으면 레거시 모델 파일 사용 (lotto_model.pt)
        if not model_path.exists():
            legacy_path = PROJECT_ROOT / "models_ai" / "trained" / folder / "lotto_model.pt"
            if legacy_path.exists():
                model_path = legacy_path
                print(f"⚠️ {lottery_id}용 모델 없음, 레거시 모델 사용: {legacy_path.name}")
            else:
                if model_type == "lstm":
                    raise HTTPException(status_code=400, detail=f"LSTM 모델이 학습되지 않았습니다. Transformer를 사용해 주세요.")
                raise HTTPException(status_code=500, detail=f"모델 파일이 없습니다: {model_path}")
        
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        config = checkpoint.get("config", {})
        
        if model_type == "transformer":
            model = create_transformer(config)
        else:
            model = create_lstm(config)
            
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        MODELS[cache_key] = model
        
    return MODELS[cache_key]


def load_draws(lottery_id: str = "korea_645") -> list:
    """로또 데이터 로드"""
    data_file = DATA_DIR / lottery_id / "draws.json"
    if not data_file.exists():
        return []
    with open(data_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("draws", [])


def load_history(lottery_id: str = "korea_645") -> list:
    """생성 이력 로드 (로또별 분리)"""
    history_file = DATA_DIR / lottery_id / "generations.json"
    if not history_file.exists():
        return []
    with open(history_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_history(history: list, lottery_id: str = "korea_645"):
    """생성 이력 저장 (로또별 분리)"""
    history_file = DATA_DIR / lottery_id / "generations.json"
    print(f"[DEBUG] 저장 시도: {history_file}, 데이터 수: {len(history)}")
    history_file.parent.mkdir(parents=True, exist_ok=True)
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] 저장 완료!")


def analyze_numbers(numbers: list) -> dict:
    """번호 분석"""
    return {
        "sum": sum(numbers),
        "odd_count": sum(1 for n in numbers if n % 2 == 1),
        "even_count": sum(1 for n in numbers if n % 2 == 0),
        "low_count": sum(1 for n in numbers if n <= 22),
        "high_count": sum(1 for n in numbers if n > 22),
    }





def compare_numbers(generated: list, winning: list) -> dict:
    """생성 번호와 당첨 번호 비교"""
    generated_set = set(generated)
    winning_set = set(winning)
    matched = generated_set & winning_set
    return {
        "matched_count": len(matched),
        "matched_numbers": sorted(list(matched)),
    }


# ========== API 엔드포인트 ==========

@app.post("/api/config")
async def save_config(config: dict):
    """설정 저장"""
    config_path = PROJECT_ROOT / "web" / "config.json"
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        print(f"[DEBUG] 설정 저장 완료: {config_path}")
        return {"status": "success", "message": "Configuration saved"}
    except Exception as e:
        print(f"[ERROR] 설정 저장 실패: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save config: {str(e)}")


class GenerateRequest(BaseModel):
    lottery_id: str = "korea_645"
    count: int = 5
    temperature: float = 1.0
    model_type: str = "transformer"  # transformer, lstm, vector
    # 필터 옵션
    ac_filter: bool = True          # AC >= 7 필터 (기본 활성화)
    sum_filter: bool = False        # 합계 100~175 필터
    consecutive_filter: bool = False # 연속번호 3개 이상 제외


class GenerateResponse(BaseModel):
    generated_at: str
    lottery_id: str
    target_draw: int
    numbers: list
    saved: bool


@app.get("/")
async def root():
    """메인 페이지"""
    return FileResponse(Path(__file__).parent.parent / "web" / "index.html")


@app.get("/api/lotteries")
async def get_lotteries():
    """사용 가능한 로또 목록"""
    config_path = Path(__file__).parent.parent / "config" / "lotteries.json"
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/draws/{lottery_id}")
async def get_draws(lottery_id: str, limit: int = 10):
    """최근 당첨 번호"""
    draws = load_draws(lottery_id)
    return draws[-limit:][::-1] if draws else []


@app.post("/api/generate")
async def generate_numbers(req: GenerateRequest):
    """AI 번호 생성"""
    model = get_model(req.model_type, req.lottery_id)
    draws = load_draws(req.lottery_id)
    
    if len(draws) < 10:
        raise HTTPException(status_code=400, detail="데이터 부족 (최소 10회차 필요)")
    
    generated = []
    
    # Vector 모델은 별도 처리 (generate() 메서드 사용)
    if req.model_type == "vector":
        # Vector가 더 많이 생성하도록 요청 (필터로 걸러질 수 있으므로)
        results = model.generate(count=req.count * 3, temperature=req.temperature)
        for r in results:
            if len(generated) >= req.count:
                break
            
            numbers = r['numbers']
            
            # 합계 필터 (100~175)
            if req.sum_filter and not is_valid_sum(numbers):
                continue
            
            # 연속번호 필터 (3개 이상 연속 제외)
            if req.consecutive_filter and has_consecutive(numbers):
                continue
            
            analysis = analyze_numbers(numbers)
            analysis["ac_value"] = r['ac_value']
            analysis["ac_rating"] = get_ac_rating(numbers)
            analysis["similarity"] = r['similarity']
            analysis["cluster"] = r['cluster']
            generated.append({
                "numbers": numbers,
                "analysis": analysis
            })

    elif req.model_type == "hot_trend":
        # Hot Trend: 최근 빈도 기반 가중치
        print("[INFO] Hot Trend Generating...")
        
        # 최근 30회차 분석 (데이터가 적으면 전체 사용)
        recent_count = min(len(draws), 30)
        recent_draws = [d["numbers"] for d in draws[-recent_count:]]
        
        frequency = Counter()
        for draw in recent_draws:
            frequency.update(draw)
            
        # 가중치 계산
        weights = {}
        total_weight = 0
        for n in range(1, 46):
            w = 1 + (frequency[n] * 2)
            weights[n] = w
            total_weight += w
            
        # 번호 생성
        for _ in range(req.count):
            current_set = set()
            attempts = 0
            while len(current_set) < 6 and attempts < 100:
                attempts += 1
                rand_val = random.uniform(0, total_weight)
                cumulative = 0
                selected = -1
                
                for n in range(1, 46):
                    cumulative += weights[n]
                    if rand_val <= cumulative:
                        selected = n
                        break
                
                if selected != -1:
                    current_set.add(selected)
            
            numbers = sorted(list(current_set))
            
            # 분석 데이터
            analysis = analyze_numbers(numbers)
            ac_val = calculate_ac(numbers)
            analysis["ac_value"] = ac_val
            analysis["ac_rating"] = get_ac_rating(numbers)
            analysis["model"] = "hot_trend"
            
            generated.append({
                "numbers": numbers,
                "analysis": analysis
            })

    else:
        # Transformer/LSTM: predict() 메서드 사용
        recent = [d["numbers"] for d in draws[-10:]]
        input_tensor = torch.tensor([recent], dtype=torch.long)
        
        seen = set()
        max_attempts = req.count * 20
        
        for attempt in range(max_attempts):
            if len(generated) >= req.count:
                break
            
            temp = req.temperature + (attempt * 0.1)
            prediction = model.predict(input_tensor, temperature=temp, top_k=20)
            numbers = sorted(prediction[0].tolist())
            numbers_tuple = tuple(numbers)
            
            if len(set(numbers)) == 6 and numbers_tuple not in seen:
                ac_value = calculate_ac(numbers)
                
                # AC 필터
                if req.ac_filter and len(numbers) == 6 and ac_value < 7:
                    continue
                
                # 합계 필터 (100~175)
                if req.sum_filter and not is_valid_sum(numbers):
                    continue
                
                # 연속번호 필터 (3개 이상 연속 제외)
                if req.consecutive_filter and has_consecutive(numbers):
                    continue
                
                seen.add(numbers_tuple)
                analysis = analyze_numbers(numbers)
                analysis["ac_value"] = ac_value
                analysis["ac_rating"] = get_ac_rating(numbers)
                generated.append({
                    "numbers": numbers,
                    "analysis": analysis
                })
    
    # 다음 회차 예측
    latest_draw = draws[-1]["draw_no"]
    target_draw = latest_draw + 1
    
    # 이력 저장
    history = load_history(req.lottery_id)
    record = {
        "id": len(history) + 1,
        "generated_at": datetime.now().isoformat(),
        "lottery_id": req.lottery_id,
        "target_draw": target_draw,
        "model": req.model_type,  # 선택된 모델 타입 저장
        "numbers": generated,
        "result": None  # 나중에 비교할 때 채워짐
    }
    history.append(record)
    save_history(history, req.lottery_id)
    
    return {
        "generated_at": record["generated_at"],
        "lottery_id": req.lottery_id,
        "target_draw": target_draw,
        "model": req.model_type,
        "numbers": generated,
        "saved": True
    }


@app.get("/api/history")
async def get_history(lottery_id: str = "korea_645", limit: int = 20):
    """생성 이력 조회"""
    history = load_history(lottery_id)
    return history[-limit:][::-1]


@app.get("/api/compare/{lottery_id}/{history_id}")
async def compare_with_result(lottery_id: str, history_id: int):
    """생성 번호와 실제 당첨 비교"""
    history = load_history(lottery_id)
    
    # 해당 이력 찾기
    record = None
    for h in history:
        if h["id"] == history_id:
            record = h
            break
    
    if not record:
        raise HTTPException(status_code=404, detail="이력을 찾을 수 없습니다.")
    
    # 당첨 결과 찾기
    draws = load_draws(record["lottery_id"])
    target_draw = record["target_draw"]
    
    winning = None
    for d in draws:
        if d["draw_no"] == target_draw:
            winning = d
            break
    
    if not winning:
        return {
            "status": "pending",
            "message": f"{target_draw}회차 결과가 아직 없습니다.",
            "record": record
        }
    
    # 비교
    comparisons = []
    for gen in record["numbers"]:
        comp = compare_numbers(gen["numbers"], winning["numbers"])
        comparisons.append({
            "generated": gen["numbers"],
            "winning": winning["numbers"],
            **comp
        })
    
    # 최고 일치
    best = max(comparisons, key=lambda x: x["matched_count"])
    
    return {
        "status": "completed",
        "target_draw": target_draw,
        "draw_date": winning["draw_date"],
        "winning_numbers": winning["numbers"],
        "bonus": winning["bonus"],
        "comparisons": comparisons,
        "best_match": best["matched_count"]
    }


# Static files & Data
STATIC_DIR = PROJECT_ROOT / "web"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/js", StaticFiles(directory=STATIC_DIR / "js"), name="js")
app.mount("/css", StaticFiles(directory=STATIC_DIR / "css"), name="css")
app.mount("/data", StaticFiles(directory=DATA_DIR), name="data")

@app.get("/")
async def read_index():
    return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
