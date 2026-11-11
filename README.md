# 어느 브랜드를, 어디에 낼까 — 서울 치킨 가맹점 폐업 분석

AI 기반 비즈니스 데이터 분석(HUFS GBT, 2026-2) 7조 팀 프로젝트.
발표 상황: **50대 예비 창업자가 서울에서 치킨 가맹점을 낸다면, 어떤 브랜드를 어디에 내야 오래 살아남는가.**

처음 보는 사람은 이 순서로 읽으면 됨:
1. 이 README: 결론, 데이터, 실행 순서
2. [인수인계.md](인수인계.md): 지금 어디까지 왔고 다음에 뭘 할지, 버린 방향과 이유
3. [data/SOURCES.md](data/SOURCES.md): 데이터 출처·이용조건 (모두 공개 데이터)
4. [AI활용로그.md](AI활용로그.md): 날짜별로 AI가 한 일과 사람이 판단한 것 (과제 제출용)

## 결론 요약

| 질문 | 답 | 근거 (스크립트) |
|---|---|---|
| 프랜차이즈가 개인 치킨집보다 오래 사나 | 예. 3년 실제 폐업 13% vs 35% (OR ≈ 0.3). 표본 정의를 바꿔도 유지 | `chicken_wide.py` |
| 어떤 브랜드가 오래 사나 | 차이 큼. 서울 3년 실제 폐업 교촌 1.1% ~ 멕시카나 36.8%. 전국 이탈률(공정위)도 같은 방향 | `chicken_brand_table.py` |
| 입지로 폐업을 미리 가를 수 있나 | 거의 못 함. 상권 정의 6종·반경 5종·이력 15년·사람/이동 특성 군집·KNN 모두 AUC 0.50–0.54 | `location_units.py`, `location_clusters.py` |
| 입지에서 건질 것 | 저녁 수요가 많은 곳엔 치킨집이 몰림(상관 0.40). 수요가 낮은 동네에선 경쟁 적은 곳 17% vs 많은 곳 34% (p<0.001) | `location_units.py` |
| 브랜드마다 맞는 입지가 있나 | 프랜차이즈는 주거지 상권에서 덜 망함(18% vs 먹자골목 26%, p=0.002). 브랜드별 궁합은 검증하면 대부분 사라짐(교촌이 어디서나 낮은 것만 확인) | `brand_fit.py`, `fit_validate.py` |
| 동 단위 시장은 어떤가 | 서울 치킨집 2021→25년 16% 감소, 매출은 2023년 정점 후 −22%. 동에서 치킨집이 1% 늘면 점포당 매출 −0.64%(포화). 다만 개별 점포 폐업은 동 매출로 못 가름 | `dong_market.py` |
| 수익과 위험을 같이 보면 | 교촌·굽네·60계가 '덜 망하고 많이 파는' 쪽. 창업비가 비싸다고 덜 망하지는 않음(순위상관 −0.14) | `brand_risk_return.py`, `figures/brand_risk_return.png` |
| 입지 변수를 전부 합치면 | 브랜드+면적 AUC 0.674에 입지 20개 변수를 더해도 0.672. 개업 후 주변 변화를 넣은 연 단위 모형도 같음. 직전 가게가 1년 못 버틴 자리만 약한 신호(32% vs 26%) | `location_max.py` |
| 예측 모델이 기준선을 넘나 | 예. 브랜드+면적+브랜드 업력 3변수 로지스틱이 2021–22 개업 평가 AUC 0.700 vs 브랜드 평균 0.655(+0.045, 95% +0.025~+0.065). 위험 상위 20% 폐업 47% vs 하위 8%. 보조지표 36개·상권 매출·변화지표를 더해도 개선 없음 | `model.py`, `indicators.py`, `trdar_test.py` |
| 매출(인기)과 폐업을 같이 보면 | 브랜드는 사분면이 80% 유지(① 교촌·bhc·60계). 상권은 폐업률이 기간마다 섞이지만, 앞 기간 매출이 높던 상권이 뒤에 덜 망함(하위 25% 12.0% vs 상위 25% 6.9%, p=0.0001) | `quadrant.py`, `figures/quadrant_trdar.png` |
| 건물·땅값 정보 | 지하 매장이 더 망함(프랜차이즈 26% vs 1층 12%, p=0.009). 공시지가·층·자리 이력을 더해도 예측 AUC +0.006(유의하지 않음). 건축물대장은 수집 중 | `parcel_fetch.py`, `parcel_test.py` |
| 그럼 어디를 추천하나 | 250m 격자에서 '경쟁 적음 + 같은 브랜드 1km 밖' 칸을 후보로 거름. 백테스트에선 통과 칸 폐업 11% vs 탈락 13%로 방향은 맞지만 유의하지 않음 → **"위험 요인을 피한 후보지"로만 제시** | `grid_screen.py` |

y(폐업)의 정의: 개업 후 3년(또는 5년) 안에 인허가상 폐업했고, 같은 자리(30m)에서 1년 안에 치킨집이 다시 등록되지 않은 경우(=점주만 바뀐 경우 제외).

## 폴더

| 경로 | 내용 |
|---|---|
| `code/` | 분석 스크립트. 각 파일 첫 줄 주석이 목적. 아래 '실행 순서'가 현재 쓰는 것 |
| `code/_참고/` | 이전 주제(카페 상권) 탐색 스크립트. 경로가 옛 구조라 그대로는 안 돌아감 |
| `data/raw/` | 원천. 저장소에 있는 것과 직접 받아야 하는 것은 아래 표 참고 |
| `data/processed/` | 집계 결과 CSV. 점포 단위 파일·pkl은 저장소에 없고 스크립트가 다시 만듦 |
| `archive/` | 폐기한 주제(카페 4분면)의 결과 |
| `인수인계.md`, `AI활용로그.md` | 진행 상황·다음 할 일, AI 사용 기록 |

## 데이터

| 데이터 | 저장소에 있나 | 받는 법 |
|---|---|---|
| 서울 일반·휴게음식점 인허가 (OA-16094, OA-16095) | ✗ (214MB+) | [OA-16094](https://data.seoul.go.kr/dataList/OA-16094/S/1/datasetView.do), [OA-16095](https://data.seoul.go.kr/dataList/OA-16095/S/1/datasetView.do)에서 CSV 받아 `data/raw/인허가/일반음식점_인허가_서울.csv`, `휴게음식점_인허가_서울.csv`로 저장 |
| 공정위 브랜드 목록·가맹점 현황·외식 통합현황 | ✅ `data/raw/ftc_*.csv` | 다시 받으려면 `ftc_brand_fetch.py`, `ftc_stats_fetch.py`, `ftc_fetch.py` (data.go.kr 키 필요) |
| 지하철 역별 시간대 승하차 2015–2026 | ✅ `data/raw/지하철/` | 다시 받으려면 `subway_fetch.py` (서울 열린데이터 키 필요) |
| 지하철 출입구·역 좌표 | ✅ `data/raw/출입구/` | |
| 서울시 상권 영역 폴리곤 1,650개 | ✅ `data/raw/상권영역/area.*` | |
| 상권 단위 유동·상주·직장인구 2021–26 | ✅ `data/raw/막시무스/상권/` | |
| 행정동 경계, 서울 생활인구 2017–22 | ✗ (3.5GB) | `python code/get_public_data.py` (로그인 불필요) |
| 상권분석서비스 행정동 점포·추정매출 2021–25 | ✗ (210MB) | [OA-22172](https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do), [OA-22175](https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do)에서 연도별 CSV를 받아 `data/raw/상권분석_행정동/점포/점포_행정동_YYYY.csv`, `추정매출/추정매출_행정동_YYYY.csv`로 저장 |
| 소상공인 상가정보 2026-06 (서울) | ✗ (302MB) | [data.go.kr 15083033](https://www.data.go.kr/data/15083033/fileData.do)에서 받아 서울만 `data/raw/막시무스/상가정보/sbiz_store_info_seoul_202606.csv`로 저장 (`mobility_feats.py`만 씀) |

API 키는 `.env`에 둠(저장소에 올리지 않음). 이미 받아 둔 데이터만 쓸 거면 키 없이 돌아감.

```
DATA_GO_KR_KEY2=...      # data.go.kr, 공정위 15125467·15110241·15165094 활용신청 필요
SEOUL_OPENAPI_KEY_A=...  # data.seoul.go.kr OpenAPI 인증키
```

## 실행 순서 (치킨 분석)

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

| 순서 | 스크립트 | 만드는 것 / 하는 일 | 필요한 원천 |
|---|---|---|---|
| 1 | `review.py` | 인허가 2종 결합 → `data/processed/_lic_all.pkl` | 인허가 |
| 2 | `chicken_wide.py` | 치킨 표본 정의 3단계(좁음·넓음·최대) 비교. 다른 스크립트가 import해서 표본으로 씀 | 1 |
| 3 | `chicken_brand_table.py` | 브랜드 비교표 → `data/processed/chicken_brand_table.csv` | 2, 공정위 |
| 4 | `mobility_feats.py` | 생활인구 동×연도, 지하철 역×연도 집계 | 생활인구, 지하철, 상가정보 |
| 5 | `location_units.py` | 지역 단위·반경·수요→경쟁→폐업 경로·이동 변수 검증 | 2, 4, 행정동 경계, 상권 폴리곤 |
| 6 | `location_clusters.py` | 긴 기간·사람/이동 특성 군집·KNN 검증 | 2, 상권 폴리곤·상권 인구 |
| 7 | `grid_screen.py` | 250m 격자 선별 백테스트 + 브랜드별 현재 후보 칸 → `data/processed/grid_candidates.csv` | 2 |
| 8 | `dong_market.py` | 상권분석서비스 동 단위 치킨 시장 현황·인허가 교차검증·포화·동 매출→폐업 → `data/processed/dong_chicken_panel.csv` | 2, 상권분석서비스 행정동, 행정동 경계 |
| 9 | `brand_risk_return.py` | 브랜드 수익·위험 지도 → `figures/brand_risk_return.png` | 3 |
| 10 | `location_max.py` | 입지 변수 전부 합친 예측 한계 + 연 단위 위험 모형 (약 6분) | 2, 상권 폴리곤, 출입구 |
| 11 | `indicators.py` | 개업 시점 보조지표 36개 생성·검증 → `indicator_tests.csv`, `_indicators.pkl` | 2, 5의 `location_units_sample.pkl`, 공정위 |
| 12 | `model.py` | 지표 선택(학습 2018–20)·예측 방정식·기준선 비교 → `model_equation.csv`, `model_eval.csv` | 9 |
| 13 | `trdar_api_fetch.py` → `trdar_test.py 3` (또는 `2`) | 상권 추정매출·상권변화지표 OpenAPI 수집, 개업 직전 분기 값 검증 → `trdar_tests_3y/2y.csv` | 2, 서울 OpenAPI 키 |

| 11 | `site_history.py` | 자리 이력(가게 회전)과 단계별 거름망 | 2 |
| 12 | `trdar_fetch.py` → `quadrant.py` | 상권 단위 치킨 매출·점포 수집 → 매출×폐업 사분면(상권·구·유형·브랜드)과 기간 검증 | 2, 서울 열린데이터 키 |
| 13 | `parcel_fetch.py` → `parcel_test.py` | 공시지가·건축물대장 필지 수집 → 필지 정보 폐업 검증 | 2, 서울 열린데이터·data.go.kr 키 |

1→2만 돌리면 3, 6, 7은 바로 돌아감(각 20초 안팎). 4, 5는 생활인구가 있어야 함.
나머지 스크립트(`fit_*`, `exit_*`, `rec_*`, `micro_feats.py` 등)는 업종 전체를 보던 이전 단계 탐색이고, 결과는 [인수인계.md](인수인계.md)의 '버린 방향'에 정리돼 있음.
