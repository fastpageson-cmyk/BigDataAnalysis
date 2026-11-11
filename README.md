# 어느 브랜드를, 어디에 낼까 — 서울 프랜차이즈 점포 폐업 분석

AI 기반 비즈니스 데이터 분석(HUFS GBT, 2026-2) 7조 팀 프로젝트.
발표 상황: 50대 예비 창업자가 서울에서 치킨 가맹점을 낸다면 어떤 브랜드를 어디에 내야 오래 살아남는가.

- 맥락·결정 기록: [인수인계.md](인수인계.md), AI 사용 기록: [AI활용로그.md](AI활용로그.md)
- 데이터 출처·이용조건: [data/SOURCES.md](data/SOURCES.md) (모두 공개 데이터)

## 폴더

| 경로 | 내용 |
|---|---|
| `code/` | 분석 스크립트. 각 파일 첫 줄 주석이 목적 |
| `code/_참고/` | 이전 주제(카페 상권) 탐색 스크립트, 경로 옛 구조 |
| `data/raw/` | 원천. 저장소에는 API로 받은 작은 파일만(공정위 3종, 지하철 승하차, 지하철 출입구) |
| `data/processed/` | 집계 결과 CSV. 점포 단위·바이너리(pkl)는 제외하고 스크립트로 재생성 |
| `archive/` | 폐기한 주제의 결과 |

## 재현

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

1. 저장소에 없는 원천을 `data/SOURCES.md`의 링크에서 받아 표의 경로에 둠: 인허가 2종, 상권분석서비스, 서울 생활인구(OA-14991, 2017–22 반기 zip), 상가정보.
   막시무스(DAT) 사본은 `code/pull_maximus.py`가 로컬 경로에서 복사함.
2. API 키는 `.env`에 둠(저장소에 올리지 않음): `DATA_GO_KR_KEY2`(공정위), `SEOUL_OPENAPI_KEY_A`(서울 열린데이터).
3. 실행 순서(치킨 분석):
   - `code/wide.py` → 인허가 결합 `_lic_all.pkl`
   - `code/chicken_wide.py` 치킨 표본 정의 3단계 비교
   - `code/ftc_brand_fetch.py`, `code/ftc_stats_fetch.py` 공정위 수집 → `code/chicken_brand_table.py` 브랜드 비교표
   - `code/subway_fetch.py` 지하철 수집 → `code/mobility_feats.py` 이동 피처 → `code/mobility_test.py` 검증

## 지금까지 결과 (치킨)

- 프랜차이즈 여부가 3년 실제 폐업을 가장 크게 가름(OR ≈ 0.3). 표본 정의를 바꿔도 유지.
- 브랜드 차이 큼: 서울 3년 실제 폐업률 교촌 1.1% ~ 멕시카나 36.8% (`data/processed/chicken_brand_table.csv`).
- 입지 변수는 약함: 경쟁 밀도는 정의에 따라 흔들리고, 이동 데이터(생활인구·지하철)는 전체 표본에서 유의하나(p=0.022) 예측 개선은 작고(AUC 0.688→0.697), 같은 브랜드 안에서는 효과 없음(p=0.74).
