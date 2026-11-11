# 막시무스(DAT) 저장소에서 이 분석에 쓸 공개 데이터를 data/raw/막시무스/로 가져온다.
# 원본은 그대로 복사, 가공본은 치킨만 추리고 개인정보·연락처 컬럼(리뷰 작성자명, 전화번호)을 뺀다.
# 제외: B078/B079(빅데이터캠퍼스 비공개), 경기도 자료(분석 범위 밖), API 캐시.
import os,shutil,zipfile,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T=f'{O}/data/raw/막시무스'; os.makedirs(T,exist_ok=True)
M='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스'; A=f'{M}/maximus-analysis-archive/data/raw'
copy={  # 행정동 시계열(2023–25가 기존 2021–22·2025–26 사이를 메움)과 인구·집객시설
 **{f'{A}/seoul_commercial_{k}_dong_{y}.csv':f'행정동/{k}_{y}.csv' for k in ['sales','stores'] for y in [2023,2024,2025]},
 f'{A}/seoul_commercial_floating_population_dong.csv':'행정동/유동인구_행정동_2021_2026.csv',
 f'{A}/seoul_resident_population_dong.csv':'행정동/상주인구_행정동.csv',
 f'{A}/seoul_workplace_population_dong.csv':'행정동/직장인구_행정동.csv',
 f'{A}/seoul_attracting_facilities_dong.csv':'행정동/집객시설_행정동.csv',
 # 상권(1,650개) 단위 인구: area_assign.py 상권 배정과 바로 결합
 f'{M}/데이터/원천/seoul_openapi/commercial_area_floating_population.csv':'상권/유동인구_상권_2021_2026.csv',
 f'{M}/데이터/원천/seoul_openapi/commercial_area_resident_population.csv':'상권/상주인구_상권.csv',
 f'{M}/데이터/원천/seoul_openapi/commercial_area_workplace_population.csv':'상권/직장인구_상권.csv',
 # 서울 목적별 생활이동(행정동 도착, 시간대·목적·성연령) 2026-08 한 달 → 귀가·여가 유입의 고정 특성
 f'{M}/데이터/원천/seoul_purpose_admdong1_in_202608.zip':'이동/seoul_purpose_admdong1_in_202608.zip',
 # 소상공인 상가정보 2026-06(층 정보 → 배달형 대리지표, 현재 점포 구성)
 f'{M}/데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv':'상가정보/sbiz_store_info_seoul_202606.csv',
}
for s,d in copy.items():
    p=f'{T}/{d}'; os.makedirs(os.path.dirname(p),exist_ok=True)
    if not os.path.exists(p) or os.path.getsize(p)!=os.path.getsize(s): shutil.copy2(s,p)
    print('복사',d,round(os.path.getsize(p)/1e6,1),'MB')
KW='치킨|통닭|닭강정|BBQ|비비큐|bhc|비에이치씨|교촌|굽네|네네|페리카나|멕시카나|처갓집|호식이|푸라닭|자담|지코바|60계|노랑통닭|또래오래|깐부'
s=pd.read_csv(f'{M}/데이터/가공/서울캠/maximus_seoul_stores_all_v0.csv',low_memory=False)
s=s[s.소분류명.astype(str).str.contains('치킨')|s.상호.astype(str).str.contains(KW,case=False)].drop(columns=['rt_전화'],errors='ignore')
os.makedirs(f'{T}/가공',exist_ok=True); s.to_csv(f'{T}/가공/치킨점포_층_평점_v0.csv',index=False,encoding='utf-8-sig'); print('치킨 점포(층·네이버평점)',len(s))
r=pd.read_csv(f'{M}/데이터/가공/naver_reviews/dining_reviews_clean.csv',low_memory=False)
r=r[r.category_small.astype(str).str.contains('치킨')|r.store_name.astype(str).str.contains(KW,case=False)]
r=r.drop(columns=['author_name'],errors='ignore'); r.to_csv(f'{T}/가공/치킨_네이버리뷰_동대문30분권.csv',index=False,encoding='utf-8-sig'); print('치킨 리뷰',len(r),'점포',r.store_name.nunique())
