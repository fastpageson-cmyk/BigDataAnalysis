# 자기잠식 지표 검토: ① 신규 점포 입장(최근접 같은 브랜드 거리 구간별 5년 폐점, 브랜드 대비) ② 기존 점포 입장(근처 300m에 같은 브랜드 신규 개점 → 이후 2년 폐점)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl')
e=s[(s.open.dt.year>=2008)&(s.open<=END-pd.DateOffset(years=5))].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=5))&e.진짜폐점).astype(int)
e['기대']=e.groupby('브랜드').y.transform('mean')
e['거리']=pd.cut(e.최근접같은브랜드m,[0,100,200,300,500,1000,1501],labels=['~100m','100-200','200-300','300-500','500m-1km','1km+'],include_lowest=True)
for g in [None,'커피','치킨']:
    x=e if g is None else e[e.업종==g]; t=x.groupby('거리',observed=True).agg(n=('y','size'),폐점률=('y','mean'),기대=('기대','mean')); t['브랜드대비']=(t.폐점률/t.기대).round(2)
    print(f'\n## ① 신규 점포: 최근접 같은 브랜드 거리 ({g or "전체"})'); print(t.round(3).to_string())
# ② 기존 점포 패널: 매년 말 영업 중인 점포, 그 해 300m 안 같은 브랜드 신규 개점 여부 → 다음 2년 실제 폐점
B=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); B=B[(B.n_match==1)&B.X.notna()&(B.브랜드!='스타벅스(직영)')].reset_index(drop=True)
T=cKDTree(B[['X','Y']].values); bo=B.open.values; bb=B.브랜드.values; bc=B.close.fillna(pd.Timestamp('2100-01-01')).values; R=[]
for Y in range(2010,2024):
    t0=np.datetime64(f'{Y}-01-01'); t1=np.datetime64(f'{Y}-12-31'); t3=np.datetime64(f'{Y+2}-12-31')
    live=np.where((bo<t0)&(bc>t1))[0]
    for i in live:
        q=np.array(T.query_ball_point([B.X[i],B.Y[i]],300),dtype=int); q=q[(q!=i)&(bb[q]==bb[i])]
        new=((bo[q]>=t0)&(bo[q]<=t1)).sum(); ex=((bo[q]<t0)&(bc[q]>t0)).sum()
        R.append((bb[i],Y,(Y-B.open[i].year),new>0,ex,int(bool(B.진짜폐점[i]) and bc[i]<=t3)))
P=pd.DataFrame(R,columns=['브랜드','연','업력','신규근접','기존근접수','y'])
P=P[P.업력>=1]
m=smf.logit('y~신규근접+np.log1p(기존근접수)+C(브랜드)+C(연)+np.log(업력)',P).fit(disp=0,method='bfgs',maxiter=500)
print(f'\n## ② 기존 점포 패널 n={len(P)} (점포-연도), 2년 내 실제폐점률 {P.y.mean():.3f}')
print(P.groupby('신규근접').y.agg(['size','mean']).round(3).to_string())
print(f'300m 안 같은 브랜드 신규 개점 → 기존 점포 2년 폐점 오즈 {np.exp(m.params["신규근접[T.True]"]):.2f}배, p={m.pvalues["신규근접[T.True]"]:.3f}')
