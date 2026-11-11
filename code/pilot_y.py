# y 정의 재검토: 인허가 폐업 = 실제 폐점 + 같은 자리 재등록(점주 교체·정정) + 인근 이전
import os,numpy as np,pandas as pd
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
s=pd.read_pickle(f'{O}/data/processed/_brand_stores.pkl'); s=s[(s.n_match==1)&(s.브랜드!='스타벅스(직영)')].copy()
s['X']=pd.to_numeric(s['좌표정보(X)'],errors='coerce'); s['Y']=pd.to_numeric(s['좌표정보(Y)'],errors='coerce')
END=pd.Timestamp('2026-11-04')
def nxt(r,g,rad):
    o=g[(g.관리번호!=r.관리번호)&(g.open>=r.close-pd.Timedelta(days=60))&(g.open<=r.close+pd.Timedelta(days=365))]
    return bool((np.hypot(o.X-r.X,o.Y-r.Y)<=rad).any())
cl=s[s.close.notna()&s.X.notna()&(s.open.dt.year>=2000)]
G={b:g for b,g in s[s.X.notna()].groupby('브랜드')}
s['같은자리재등록']=False; s['인근이전']=False
for i,r in cl.iterrows():
    g=G[r.브랜드]
    if nxt(r,g,30): s.at[i,'같은자리재등록']=True
    elif nxt(r,g,1000): s.at[i,'인근이전']=True
s['진짜폐점']=s.close.notna()&~s.같은자리재등록&~s.인근이전
s.to_pickle(f'{O}/data/processed/_brand_stores_y.pkl')
k=s[s.open.dt.year.between(2018,2023)&s.X.notna()]
for h in [1,2,3]:
    ok=k.open<=END-pd.DateOffset(years=h); kk=k[ok]; w=kk.close<=kk.open+pd.DateOffset(years=h)
    print(f'{h}년 n={len(kk)} 인허가폐업={w.sum()}({w.mean():.3f}) 같은자리재등록={(w&kk.같은자리재등록).sum()} 인근이전={(w&kk.인근이전).sum()} 진짜폐점={(w&kk.진짜폐점).sum()}({(w&kk.진짜폐점).mean():.3f})')
w=k.close<=k.open+pd.DateOffset(years=2); d=(k.close-k.open).dt.days
print('2년내 같은자리재등록 중 90일내',(w&k.같은자리재등록&(d<=90)).sum(),' 진짜폐점 중 90일내',(w&k.진짜폐점&(d<=90)).sum())
