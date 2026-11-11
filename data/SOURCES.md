# 데이터 출처

모두 공공누리 제1유형(출처 표시, 상업 이용·변경 허용). 수집일은 이전 세션 다운로드일.

| 파일 | 데이터명 | 제공기관 | URL | 수집일 |
|---|---|---|---|---|
| raw/인허가/휴게음식점_인허가_서울.csv | 서울시 휴게음식점 인허가 정보 (OA-16095) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-16095/S/1/datasetView.do | 2026-11-05 |
| raw/인허가/일반음식점_인허가_서울.csv | 서울시 일반음식점 인허가 정보 (OA-16094) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-16094/S/1/datasetView.do | 2026-11-05 |
| raw/상권_행정동/sales_*.csv | 서울시 상권분석서비스(추정매출-행정동) (OA-22175) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do | 2026-11-06 |
| raw/상권_행정동/stores_*.csv | 서울시 상권분석서비스(점포-행정동) (OA-22172) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do | 2026-11-06 |
| raw/상권분석_행정동/점포/점포_행정동_2021–2025.csv | 서울시 상권분석서비스(점포-행정동) (OA-22172) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-22172/S/1/datasetView.do | 2026-11-11 (팀원 제공 zip) |
| raw/상권분석_행정동/추정매출/추정매출_행정동_2021–2025.csv | 서울시 상권분석서비스(추정매출-행정동) (OA-22175) | 서울특별시 | https://data.seoul.go.kr/dataList/OA-22175/S/1/datasetView.do | 2026-11-11 (팀원 제공 zip) |
| raw/상권_분기/치킨_매출_상권.csv, 치킨_점포_상권.csv | 서울시 상권분석서비스(추정매출-상권 OA-15572, 점포-상권 OA-15577) OpenAPI, 치킨전문점만 | 서울특별시 | https://data.seoul.go.kr/dataList/OA-15572/S/1/datasetView.do | 2026-11-11 (API) |
| raw/필지/jiga.csv | 서울시 개별공시지가 정보 (OA-1180) OpenAPI | 서울특별시 | https://data.seoul.go.kr/dataList/OA-1180/S/1/datasetView.do | 2026-11-11 (API) |
| raw/필지/brtitle.csv | 국토교통부 건축HUB 건축물대장정보 서비스 표제부 (15134735) | 국토교통부 | https://www.data.go.kr/data/15134735/openapi.do | 수집 중 |
| (DAT) seoul_commercial_*_dong*.csv, 인구·집객시설 | 서울시 상권분석서비스 행정동 시리즈 | 서울특별시 | 위 OA 시리즈 | DAT 프로젝트 수집분, 날짜 재확인 필요 |
| (DAT) sbiz_store_info_seoul_202606.csv | 소상공인시장진흥공단 상가(상권)정보 | 소상공인시장진흥공단 | https://www.data.go.kr/data/15083033/fileData.do | 2026-06 기준, 수집일 재확인 필요 |
| raw/ftc_eatout_2017_2025.csv | 공정위 가맹정보 브랜드 가맹사업자 통합정보 외식업종 현황 (15165094) | 공정거래위원회 | https://www.data.go.kr/data/15165094/openapi.do | 2026-11-08 (API) |
| raw/상권영역/area.*, 상권_추정매출_2025.csv | 서울시 상권분석서비스(영역-상권), (추정매출-상권) 2025 | 서울특별시 | DAT 저장소 commercial_area/*.zip 사본, 원 출처 data.seoul.go.kr | DAT 수집분, 날짜 재확인 필요 |
| raw/출입구/osm_subway_entrance_seoul.json | OpenStreetMap 서울 지하철 출입구(railway=subway_entrance, ref=출구번호) | OpenStreetMap 기여자 | https://overpass-api.de (Overpass API) | 2026-11-08, ODbL(출처 표기 필요) |
| raw/출입구/seoul_station_master_20260902.csv | 서울시 역사마스터 정보(역 좌표) | 서울특별시 | DAT 저장소 사본, 원 출처 data.seoul.go.kr | 2026-09-02 |
| raw/도로망/osm_roads_seoul.json | OpenStreetMap 서울 도로(highway 12종, 서울 경계 상자) | OpenStreetMap 기여자 | https://overpass-api.de (Overpass API) | 2026-11-08, ODbL(출처 표기 필요) |
| raw/행정동경계/HangJeongDong_ver20200701.geojson | 행정동 경계 2020-07 (EPSG:5179), 통계청 SGIS 경계 기반 가공 | admdongkor(vuski), CC BY 4.0 | https://github.com/vuski/admdongkor | 2026-11-11 |
| (추가 후보) 전국 일반음식점 인허가 | 행정안전부_식품_일반음식점 (15045016) | 행정안전부 | https://www.data.go.kr/data/15045016/fileData.do | 미수집 |
| raw/ftc_brand_list_2017_2025.csv | 공정위 가맹정보 브랜드 목록 정보 (15125467), 대표자명 제외 저장 | 공정거래위원회 | https://www.data.go.kr/data/15125467/openapi.do | 2026-11-11 (API) |
| raw/ftc_brand_frcs_stats_2017_2025.csv | 공정위 가맹정보 브랜드별 가맹점 현황 (15110241) | 공정거래위원회 | https://www.data.go.kr/data/15110241/openapi.do | 2026-11-11 (API) |
| raw/생활인구/LOCAL_PEOPLE_DONG_*.zip | 서울 생활인구(행정동) 내국인 (OA-14991), 2017–2022 반기 파일 | 서울특별시·KT | https://data.seoul.go.kr/dataList/OA-14991/S/1/datasetView.do | 2026-11-11 |
| raw/막시무스/행정동/*.csv | 서울시 상권분석서비스 행정동 매출·점포 2023–25, 유동·상주·직장인구, 집객시설 | 서울특별시 | OA-22175·OA-22172 등 상권분석서비스 시리즈 | DAT 수집분(2026-09) 사본 |
| raw/막시무스/상권/*.csv | 서울시 상권분석서비스(상권) 유동·상주·직장인구 | 서울특별시 | data.seoul.go.kr 상권분석서비스 OpenAPI | DAT 수집분(2026-10-01) 사본 |
| raw/막시무스/이동/seoul_purpose_admdong1_in_202608.zip | 수도권 생활이동(목적별, 행정동 도착) 내국인 2026-08 (OA-22298) | 서울특별시·KT | https://data.seoul.go.kr/dataList/OA-22298/F/1/datasetView.do | DAT 수집분(2026-09) 사본 |
| raw/막시무스/상가정보/sbiz_store_info_seoul_202606.csv | 소상공인 상가(상권)정보 서울 2026-06 | 소상공인시장진흥공단 | https://www.data.go.kr/data/15083033/fileData.do | DAT 수집분 사본 |
| raw/막시무스/가공/*.csv | 막시무스 가공본 치킨 부분집합(점포 층·네이버 평점, 동대문 30분권 네이버 리뷰), 작성자명·전화번호 제외 | 김건우(DAT) 가공 | code/pull_maximus.py | 2026-11-11 |
| raw/지하철/subway_time_2015_2026.csv | 서울시 지하철 호선별 역별 시간대별 승하차 인원 (OA-12252, OpenAPI CardSubwayTime) 2015-01~2026-10 | 서울특별시 | https://data.seoul.go.kr/dataList/OA-12252/S/1/datasetView.do | 2026-11-11 (API, code/subway_fetch.py) |
| (추가 후보) 제과점영업 인허가 | 서울시 제과점영업 인허가 정보 | 서울특별시 | data.seoul.go.kr에서 "제과점영업" 검색 | 미수집 |

수집일 2026-11-05/06은 이전 세션 기록 기준. 제출 전 원본 페이지에서 갱신일·이용조건 다시 확인.
