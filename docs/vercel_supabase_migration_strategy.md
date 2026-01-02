# 🚀 Vercel + Supabase 전환 및 모바일 앱 확장 전략

본 문서는 기존의 Python FastAPI 기반 아키텍처에서 Serverless 환경인 Vercel과 Supabase로 전환하여 유지보수 비용을 절감하고 모바일 앱 확장을 실현하기 위한 상세 전략을 담고 있습니다.

---

## 🏗️ 1. 목표 아키텍처 개요 (Serverless & Edge)

기존의 상시 가동형 파이썬 서버를 제거하고, 클라이언트(브라우저/앱)가 직접 Supabase와 통신하며 복잡한 로직은 Edge 환경에서 처리하는 구조입니다.

| 구분 | AS-IS (현재) | TO-BE (전략) |
| :--- | :--- | :--- |
| **Frontend** | Vanilla JS / Simple HTML | **Quasar Framework (Vue 3)** |
| **Backend** | Python FastAPI (Local/Cloud) | **Supabase Edge Functions** |
| **Database** | Local JSON Files | **Supabase PostgreSQL** |
| **AI Inference** | Python Torch / ONNX Runtime | **Client-side ONNX Runtime** |
| **Hosting** | Local Server / Python Host | **Vercel** |

---

## 📊 2. 데이터 마이그레이션 전략 (Local JSON → DB)

JSON 파일 기반의 데이터를 관계형 데이터베이스(PostgreSQL)로 전환하여 실시간성과 쿼리 효율을 확보합니다.

### 데이터 구조 설계
- **`draws` 테이블**: 모든 국가의 당첨 번호를 통합 관리.
    - `id`: 고유 ID
    - `lottery_id`: `korea_645`, `usa_powerball` 등으로 구분
    - `draw_number`: 회차
    - `numbers`: 당첨 번호 (JSONB 또는 Array 타입)
    - `is_bonus`: 보너스 번호 여부
- **`user_history` 테이블**: 사용자가 생성한 번호 및 분석 결과 저장.

### 마이그레이션 도구
- Supabase의 **CSV Import** 기능을 사용하여 기존 데이터를 대량 업로드.
- 이후 새로운 회차 자동 갱신은 Supabase Edge Functions을 이용한 스케줄링(Cron)으로 처리.

---

## 🧠 3. 분석 모델 통합 및 실행 전략

### ① 딥러닝 모델 (Transformer, LSTM, Vector)
- **실행**: 사용자 브라우저(Web) 또는 기기(App)에서 **ONNX Runtime Web**을 통해 실행.
- **포팅**: 서버를 거치지 않으므로 대기 시간이 0ms에 가깝고 서버 비용이 발생하지 않음.
- **모델 관리**: Supabase Storage 또는 Vercel의 `public` 폴더에 국가별 `.onnx` 파일 저장.

### ② 통계 모델 (Hot/Cold, Physics, Balanced)
- **실행**: **Supabase Edge Functions (Deno/TypeScript)**.
- **이유**: Python의 `numpy` 로직을 TypeScript로 포팅하여 Edge에서 실시간 계산.
- **보안**: 알고리즘 소스 코드가 클라이언트에 노출되지 않도록 서버측에서 처리.

---

## 📱 4. Quasar(`web-vue`) 개발 및 배포 전략

### 통합 개발 환경
- 하나의 소스 코드로 웹과 모바일 앱(Capacitor) 동시 대응.
- `src/composables/useSupabase.ts`를 구현하여 전역에서 DB 접근 및 인증 관리.

### Vercel 배포 흐름
1.  GitHub 저장소와 Vercel 연동.
2.  `main` 브랜치 푸시 시 자동 빌드 및 배포 (`quasar build`).
3.  환경 변수(`SUPABASE_URL`, `SUPABASE_ANON_KEY`) 설정.

---

## 🔑 5. 단계별 전환 로드맵

### [Phase 1] 인프라 설정 (진행 중)
- [ ] Supabase 프로젝트 생성 및 테이블 스키마 설정.
- [ ] Vercel 프로젝트 생성 및 GitHub 연동.

###- [x] Phase 4: Component Development (Slot Machine, SVG Charts)
- [ ] Phase 5: ONNX AI Engine Integration (In-Progress: Troubleshooting WASM/MJS in Vite)
- [ ] Phase 6: Mobile optimization and PWA

### Phase 5 Detailed Learnings (Critical for Gemini Pro)
> [!IMPORTANT]
> Vite struggles with `.mjs` assets in the `public/` folder. For `onnxruntime-web` to work in dev mode:
> 1. Rename `.mjs` to `.js` in `public/wasm/`.
> 2. Set `ort.env.wasm.wasmPaths` to `window.location.origin + '/wasm/'`.
> 3. Add `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` to `devServer` headers.

### [Phase 2] 데이터 마이그레이션
- [ ] Python 스크립트를 작성하여 기존 `.json` 데이터를 Supabase DB로 전송.
- [ ] 최신 회차 정보를 읽어오는 크롤러를 Supabase Edge Functions으로 이식.

### [Phase 3] 프론트엔드 포팅
- [ ] `web-vue`(Quasar) 프로젝트에 현재의 UI/UX 디자인 적용.
- [ ] Vanilla JS 로직을 Vue Component로 마이그레이션.
- [ ] ONNX 모델 로딩 로직 고도화 (국가별 파일 선택 로드).

### [Phase 4] 최적화 및 앱 출시
- [ ] Lighthouse 성능 측정 및 최적화.
- [ ] Capacitor를 이용한 Android/iOS 앱 빌드 및 테스트.

---

## 💡 기대 효과
- **비용 절감**: 모든 서비스가 Free Tier 내에서 운영 가능 (대규모 트래픽 발생 전까지 비용 0원).
- **운영 편의성**: 서버 관리 부담이 없어지며, DB 수정만으로 앱 전체 데이터 즉시 업데이트.
- **사용자 경험**: 모바일 네이티브 앱과 같은 부드러운 애니메이션과 빠른 AI 분석 속도 제공.
