# 상권을 미리 정의하지 않고 점포 반경 300m의 "주변 구성"을 피처로 만든다
#  (가) 인허가(개업 시점 기준, 시점 일치): 업태별 영업 중 음식점 수, 1년 개·폐업, 최근 3년 폐업 점포의 평균 수명, 프랜차이즈 비중
#  (나) 소상공인 상가정보 2026-06(현재 스냅숏, 시점 불일치 표시): 업종 구성 비중
import os,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=300
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl')
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl')
a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce'); a=a[a.X.notna()&a.open.notna()].reset_index(drop=True)
G={'한식':['한식','식육(숯불구이)','탕류(보신용)','횟집','뷔페식'],'분식김밥':['분식','김밥(도시락)'],'양식일식중식':['경양식','일식','중국식','외국음식전문점(인도,태국등)'],
   '카페':['커피숍','까페','다방','전통찻집'],'술집':['호프/통닭','정종/대포집/소주방','감성주점'],'치킨':['통닭(치킨)'],'패스트푸드':['패스트푸드'],'제과':['과자점','아이스크림'],'편의점':['편의점']}
cat=np.full(len(a),'기타',dtype=object)
for k,v in G.items(): cat[a.업태구분명.isin(v).values]=k
ao=a.open.values; ac=a.close.fillna(pd.Timestamp('2100-01-01')).values; life=(a.close-a.open).dt.days.values/365.25
fr=a.관리번호.isin(pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl').관리번호).values
T=cKDTree(a[['X','Y']].values); rows=[]
for r in s.itertuples():
    t=np.datetime64(r.open); p=np.array(T.query_ball_point([r.X,r.Y],R),dtype=int); live=(ao[p]<t)&(ac[p]>t); n=max(live.sum(),1)
    d={f'주변_{k}비중':(live&(cat[p]==k)).sum()/n for k in G}
    c3=(ac[p]<=t)&(ac[p]>t-np.timedelta64(3*365,'D'))
    d.update(주변_음식점수=live.sum(),주변_1y폐업률=((ac[p]<=t)&(ac[p]>t-np.timedelta64(365,'D'))).sum()/n,주변_1y개업률=((ao[p]<=t)&(ao[p]>t-np.timedelta64(365,'D'))).sum()/n,
             주변_폐업점평균수명=np.nanmean(life[p][c3]) if c3.any() else np.nan,주변_프랜차이즈비중=(live&fr[p]).sum()/n)
    rows.append(d)
F=pd.DataFrame(rows,index=s.index)
# (나) 상가정보 스냅숏
sb=pd.read_csv('/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv',usecols=['상권업종대분류명','상권업종중분류명','경도','위도'],low_memory=False).dropna(subset=['경도'])
sb['X'],sb['Y']=Transformer.from_crs(4326,5174,always_xy=True).transform(sb.경도.values,sb.위도.values)
M={'사무':['본사·경영 컨설팅','광고','전문 디자인','기술 서비스','법무관련 ','회계·세무','기타 사업 서비스'],'학원':['일반 교육','기타 교육','교육 지원'],'병의원':['의원'],
   '부동산':['부동산 서비스'],'의류쇼핑':['섬유·의복·신발 소매','의약·화장품 소매'],'오락':['유원지·오락'],'숙박':['일반 숙박','기타 숙박'],'생활서비스':['이용·미용','세탁','식료품 소매'],'주점':['주점']}
sc=np.full(len(sb),'기타',dtype=object)
for k,v in M.items(): sc[sb.상권업종중분류명.isin(v).values]=k
TS=cKDTree(sb[['X','Y']].values); rows=[]
for r in s.itertuples():
    p=np.array(TS.query_ball_point([r.X,r.Y],R),dtype=int); n=max(len(p),1)
    d={f'현재_{k}비중':(sc[p]==k).sum()/n for k in M}; d['현재_업소수']=len(p); rows.append(d)
F=F.join(pd.DataFrame(rows,index=s.index))
F['주변_음식점수']=np.log1p(F.주변_음식점수); F['현재_업소수']=np.log1p(F.현재_업소수)
F.to_pickle(f'{O}/data/processed/micro_feats.pkl'); print(F.shape); print(F.describe().T[['mean','50%']].round(3).to_string())
