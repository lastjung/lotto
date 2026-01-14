# Mobile 대응 전략 (web-vue)

## 목표
- 모든 로또 지원을 유지하면서 모바일 앱 경험에 가까운 UI/UX 제공.
- PWA 기반으로 빠르게 배포하고, 필요 시 Capacitor로 확장.

## 현재 병목 (코드 기준)
- 로또 메타데이터 하드코딩으로 API/데이터와 불일치 가능.
- JSON 폴백 정렬 키가 다르며(draw_number vs draw_no) 데이터 순서가 틀어질 수 있음.
- ONNX 모델 경로 규칙이 실제 public 경로와 불일치하여 로딩 실패 위험.
- 로또 변경 시 전체 reload 사용으로 모바일 UX 저하.

## 방향성
1) PWA 우선
- 홈스크린 설치형 경험 제공.
- ONNX/wasm 캐시 전략으로 오프라인 안정성 확보.

2) Capacitor(옵션)
- 앱스토어 배포 필요 시 PWA 안정화 후 전환.
- public 자산 번들 포함 방식 유지.

## 구체적 실행 방법

### 1. 데이터 소스 통합 (API 기반)
- /api/lotteries로 로또 목록을 동적 로드.
- /api/draws/{lottery_id}로 draw 데이터 로드.
- JSON 폴백은 draw_no/draw_number 호환 정렬 로직 포함.

### 2. 모델 자산 경로 통일
- 권장 경로: public/models/{model}/{lottery_id}.onnx
- 로더는 위 구조만 사용하도록 단순화.
- wasm 파일은 public/wasm 고정.

### 3. 모바일 레이아웃 재구성
- q-header + q-footer(q-tabs) 기반 하단 탭 네비.
- IndexPage는 결과 영역 중심, 설정/차트는 접기(accordion) 처리.
- 로또 변경 시 reload 제거, 상태만 업데이트.

### 4. PWA 설정
- quasar mode add pwa
- manifest 아이콘/색상 정의
- Workbox runtime caching에 *.onnx, *.wasm 추가

## 단계별 진행 플로우
1) useLotto를 API 기반으로 변경 + JSON 폴백 정렬 호환
2) 모델 경로 규칙 통일 및 public 구조 정리
3) 모바일 레이아웃 전환 (탭 네비 + 결과 중심)
4) PWA 모드 활성화 및 캐시 전략 적용

## 진행 상태 (적용 완료)
- 모바일 네비: 하단 탭(q-footer/q-tabs) 추가, 모바일에서 결과 영역이 먼저 보이도록 순서 변경.
- 차트 접근: 모바일에서는 Charts 섹션을 접이식으로 노출.
- 로또 변경: 전체 reload 제거, 상태만 갱신하도록 변경.
- PWA 설정: manifest 메타 및 ONNX/wasm 캐시 전략(workbox runtime caching) 추가.
- PWA 모드: quasar PWA 모드 추가 및 src-pwa 생성 완료.
- 실행 확인: `quasar dev -m pwa`는 PATH에 quasar가 없어 실패함. 필요 시 `npx quasar dev -m pwa`로 실행 권장.

## 다음 액션 제안
- 위 플로우 1번부터 순차적으로 적용 후, 실제 모바일 기기에서 동작 확인.
- PWA 적용 후 안정화되면 Capacitor 전환 검토.
