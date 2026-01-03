# 🎱 Lottery Data System Guide

본 문서는 로또 데이터 수집, 관리, 그리고 Supabase 동기화 시스템(`update_drawdata_db.py`)에 대한 상세 가이드입니다.

## 1. 시스템 개요 (System Overview)

이 시스템은 전 세계 주요 복권(한국, 미국, 캐나다, 일본)의 당첨 번호를 **수집(Collect)**하고, 이를 중앙 데이터베이스인 **Supabase(Star DB)**에 **동기화(Sync)**하는 역할을 합니다.

### 핵심 구성 요소
- **Collector (수집기)**: 각 국가별 로또 사이트나 API에서 데이터를 가져오는 파이썬 클래스 (`collectors/`).
- **Local Storage**: 수집된 데이터는 먼저 로컬 파일(`data/{lottery_id}/data.json`)에 저장됩니다. (1차 백업)
- **Hybrid Sync Script**: `scripts/update_drawdata_db.py`는 로컬 저장소와 원격 DB(Supabase)를 연결합니다.

---

## 2. 주요 기능 (Key Features)

### 2.1 하이브리드 수집 (Hybrid Collection)
- **과거 데이터**: SQLite DB나 공공 데이터 포털(NY Data)에서 대량의 과거 데이터를 고속으로 로드합니다.
- **최신 데이터**: 공식 웹사이트를 실시간 스크래핑하여 최신 회차를 채워 넣습니다.
- **장점**: 속도와 최신성 두 마리 토끼를 잡습니다.

### 2.2 자동 DB 동기화 (Auto DB Sync)
- 수집된 데이터는 즉시 Supabase의 `lotto_history` 테이블로 업로드됩니다.
- **중복 방지**: `upsert` 방식을 사용하여 이미 존재하는 회차는 건너뛰거나 갱신합니다.
- **안전성**: 배치(Batch) 처리로 대량 데이터도 안정적으로 전송합니다.

---

## 3. 사용법 (Usage)

### 3.1 사전 준비 (.env 설정)
프로젝트 루트에 `.env` 파일이 있어야 하며, Supabase 접속 정보가 포함되어야 합니다.

```bash
SUPABASE_URL="https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY="eyJh..." 
```

### 3.2 명령어 실행
터미널에서 다음 명령어로 실행합니다.

**모든 로또 업데이트:**
```bash
python scripts/update_drawdata_db.py
```

**특정 로또만 업데이트:**
```bash
# 캐나다 6/49
python scripts/update_drawdata_db.py --lottery canada_649

# 미국 파워볼
python scripts/update_drawdata_db.py --lottery usa_powerball
```

**전체 이어받기가 아닌 처음부터 다시 수집 (Full Refresh):**
```bash
python scripts/update_drawdata_db.py --full
```

---

## 4. 지원되는 로또 (Supported Lotteries)

| ID | 이름 | 수집 방식 | 상태 |
| :--- | :--- | :--- | :--- |
| `canada_649` | 🇨🇦 Canada 6/49 | SQLite (History) + WCLC Scraping (Recent) | ✅ Active |
| `usa_powerball` | 🇺🇸 USA Powerball | NY Data.gov API | ✅ Active |
| `usa_megamillions` | 🇺🇸 USA Mega Millions | NY Data.gov API | ✅ Active |
| `japan_loto6` | 🇯🇵 Japan Loto 6 | GitHub Repo (Open Data) | ✅ Active |
| `korea_645` | 🇰🇷 Korea 6/45 | DhLottery API | ⚠️ Blocked (VPN 필요) |

---

## 5. 데이터베이스 스키마 (DB Schema)

Supabase의 `lotto_history` 테이블 구조입니다.

| 컬럼명 | 타입 | 설명 |
| :--- | :--- | :--- |
| `lottery_id` | `text` | 로또 식별자 (예: `canada_649`) |
| `draw_no` | `int4` | 회차 번호 |
| `draw_date` | `date` | 추첨일 |
| `numbers` | `int4[]` | 당첨 번호 배열 (보너스 제외) |
| `bonus` | `int4` | 보너스 번호 (nullable) |

---

## 6. 문제 해결 (Troubleshooting)

- **Supabase 연결 실패**: `.env` 파일의 KEY가 정확한지 확인하세요. `SERVICE_ROLE_KEY`를 사용해야 쓰기 권한이 확실합니다.
- **컬럼 없음 에러**: Supabase 대시보드 SQL Editor에서 테이블 생성 스크립트(Table Create SQL)를 다시 실행하세요.
- **Connection Timeout**: 한국 로또 등 일부 사이트는 해외 IP를 차단할 수 있습니다. 로컬 환경의 네트워크 상태를 확인하세요.
