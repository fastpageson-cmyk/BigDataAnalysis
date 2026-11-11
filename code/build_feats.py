# 브랜드 점포(2008~2025 개업) 개업 시점 변수 표 생성 → data/processed/store_feats.pkl
import os,numpy as np,pandas as pd
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl')
a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce'); a=a[a.X.notna()&a.open.notna()].reset_index(drop=True)
T=cKDTree(a[['X','Y']].values); ao=a.open.values; ac=a.close.fillna(pd.Timestamp('2100-01-01')).values; acafe=(a.src=='cafe').values
B=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); B=B[(B.n_match==1)&B.X.notna()].reset_index(drop=True)
TB=cKDTree(B[['X','Y']].values); bo=B.open.values; bc=B.close.fillna(pd.Timestamp('2100-01-01')).values; bb=B.브랜드.values; bg=B.업종.values; bid=B.관리번호.values
s=B[(B.브랜드!='스타벅스(직영)')&B.open.dt.year.between(2008,2025)].copy().reset_index(drop=True)
R=[]
for r in s.itertuples():
    t=np.datetime64(r.open); t1=t-np.timedelta64(365,'D')
    p=np.array(T.query_ball_point([r.X,r.Y],300),dtype=int); live=(ao[p]<t)&(ac[p]>t)
    p5=np.array(T.query_ball_point([r.X,r.Y],500),dtype=int)
    q=np.array(TB.query_ball_point([r.X,r.Y],500),dtype=int); q=q[bid[q]!=r.관리번호]; lb=(bo[q]<t)&(bc[q]>t)
    q1=np.array(TB.query_ball_point([r.X,r.Y],1000),dtype=int); q1=q1[bid[q1]!=r.관리번호]; lb1=(bo[q1]<t)&(bc[q1]>t)
    R.append(dict(밀도300=live.sum(),휴게밀도300=(live&acafe[p]).sum(),폐업1y500=((ac[p5]<=t)&(ac[p5]>t1)).sum(),신규1y500=((ao[p5]<=t)&(ao[p5]>t1)).sum(),
        같은브랜드500=(lb&(bb[q]==r.브랜드)).sum(),같은브랜드1km=(lb1&(bb[q1]==r.브랜드)).sum(),같은업종타브랜드500=(lb&(bg[q]==r.업종)&(bb[q]!=r.브랜드)).sum(),
        같은업종신규1y500=((bo[q]<=t)&(bo[q]>t1)&(bg[q]==r.업종)&(bb[q]!=r.브랜드)).sum(),
        최근접같은브랜드m=np.min(np.hypot(B.X.values[q1[lb1&(bb[q1]==r.브랜드)]]-r.X,B.Y.values[q1[lb1&(bb[q1]==r.브랜드)]]-r.Y)) if (lb1&(bb[q1]==r.브랜드)).any() else 1500))
s=pd.concat([s,pd.DataFrame(R)],axis=1)
s['면적']=np.log1p(pd.to_numeric(s['소재지면적'],errors='coerce').fillna(55)); s['휴게']=(s.src=='cafe').astype(int); s['다중']=(s['다중이용업소여부']=='Y').astype(int)
s['구']=s['지번주소'].str.extract(r'서울특별시\s+(\S+구)')[0].fillna('NA'); s['연']=s.open.dt.year; s['분기']=s.open.dt.quarter
# 동 인허가 변수: 개업 직전 연말
dl=pd.read_pickle(f'{O}/data/processed/_dong_lic_yearly.pkl'); s['Yp']=s.연-1; s['adm']=s.adm.astype('Int64')
s=s.merge(dl.rename(columns={'Y':'Yp'}),how='left',on=['adm','Yp'])
# 동 상권 변수: 개업 직전 분기, 2021Q1 이전 개업은 2021Q1 값(플래그)
dm=pd.read_pickle(f'{O}/data/processed/_dong_market_q.pkl')
pq=(s.open-pd.DateOffset(months=3)); yq=(pq.dt.year*10+pq.dt.quarter).clip(lower=20211); s['상권시점일치']=(pq.dt.year*10+pq.dt.quarter>=20211).astype(int); s['yq']=yq
s=s.merge(dm.drop(columns=['동_외식매출']),how='left',on=['adm','yq'])
for c in ['동_외식점포','동_점포당매출','동_유동인구','동_직장인구','동_상주인구','동_집객시설','동_음식점수']: s[c]=np.log1p(s[c])
s.to_pickle(f'{O}/data/processed/store_feats.pkl'); print(s.shape); print(s.filter(like='동_').isna().mean().round(3).to_dict())
