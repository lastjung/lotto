# 🚀 Vercel 배포 가이드 (Vercel Deployment Guide)

본 문서는 `web-static` 폴더 기반의 정적 웹사이트를 Vercel에 안전하게 배포하기 위한 설정 가이드입니다.

## 1. 프로젝트 구조 이해

현재 프로젝트는 다음과 같은 이중 구조를 가지고 있습니다.

- **`web/`**: 로컬 개발(Local Development)용. 심볼릭 링크를 통해 데이터를 참조.
- **`web-static/`**: **실제 배포(Production)**용. `index.html`, `js`, `models` 등이 모여있는 루트 폴더.

서버 없이 동작하는 **Static Site**이므로, Vercel의 정적 호스팅 기능을 사용합니다.

---

## 2. Vercel 프로젝트 설정 (필수)

Vercel 대시보드에서 프로젝트를 생성하거나 `Settings` 탭으로 이동하여 아래 두 가지를 반드시 설정해야 합니다.

### ① Root Directory (루트 디렉토리) 설정
Vercel이 "어떤 폴더를 웹사이트의 시작점(Root)으로 볼 것인가"를 정합니다.

- **위치**: `Settings` -> `General` -> `Root Directory`
- **설정값**: `web-static`
- **이유**: `index.html`이 이 폴더 안에 있기 때문입니다.

### ② Build Command (빌드 명령어) 설정
`web-static/data` 폴더가 **심볼릭 링크(바로가기)**로 되어 있어, 그냥 배포하면 데이터 파일이 누락될 수 있습니다. 이를 해결하기 위해 원본 파일을 복사하는 명령어를 입력합니다.

- **위치**: `Settings` -> `General` -> `Build & Development Settings`
- **설정값**: `Override` 체크 후 아래 명령어 입력
  ```bash
  # 1. Clean and Prepare
  rm -rf config css data js index.html
  
  # 2. Copy Assets
  cp -RL ../config ../data ../web/css ../web/js ../web/index.html .
  
  # 3. Generate Config from Env Vars
  echo "window.SUPABASE_CONFIG = { url: '${VITE_SUPABASE_URL}', key: '${VITE_SUPABASE_ANON_KEY}' };" > js/config.js
  ```
- **해석**:
  - `cp`: 복사(Copy) 명령어
  - `-R`: 폴더 내부까지 재귀적으로(Recursive)
  - `-L`: 심볼릭 링크를 따라가서 **원본 파일(Target)**을 복사(Dereference Link)
  - `../data`: 프로젝트 최상위의 원본 데이터 폴더
  - `.`: 현재 위치(`web-static`)로 복사

---

## 3. 환경 변수 (Environment Variables)

Supabase 연동을 위해 필요한 키값들을 설정합니다. (프론트엔드 코드에 하드코딩 되어 있다면 생략 가능하지만, 보안상 권장됩니다.)

- **위치**: `Settings` -> `Environment Variables`
- **변수명**:
  - `VITE_SUPABASE_URL`: Supabase 프로젝트 URL
  - `VITE_SUPABASE_ANON_KEY`: Supabase 공개 키 (Anon Key)

---

## 4. 배포 확인 (Verification)

1. 설정을 마친 후 **Deployments** 탭에서 `Redeploy`를 누릅니다.
2. 배포 로그(Build Logs)에서 `cp -RL ../data .` 명령어가 실행되는지 확인합니다.
3. 배포된 사이트 접속 후, 개발자 도구(F12) -> Console에서 다음 로그가 뜨는지 확인합니다.
   - `✅ 로또 데이터 로드 완료 (korea_645): ...회차`
   - 만약 Supabase 연동이 되어 있다면 `[Source: Supabase Cloud]`라고 뜹니다.

---

## 5. 요약 (Cheat Sheet)

| 설정 항목 | 값 (Value) | 비고 |
| :--- | :--- | :--- |
| **Framework Preset** | `Other` | 자동 감지 실패 시 선택 |
| **Root Directory** | `web-static` | |
| **Build Command** | `rm -rf config css data js index.html && cp -RL ../config ../data ../web/css ../web/js ../web/index.html .` | 심볼릭 링크 해결용 |
| **Output Directory** | (비워둠) 또는 `.` | |
