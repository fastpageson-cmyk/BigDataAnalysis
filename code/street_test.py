# 길 피처가 같은 브랜드 안에서 5년 실제폐점과 관련 있나(브랜드·연도 고정효과), 업종별 + 대로변 여부 구간표
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/street_feats.pkl'))
e=s[(s.open.dt.year>=2008)&(s.open<=END-pd.DateOffset(years=5))].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=5))&e.진짜폐점).astype(int)
F=['길_접한최고위계','길_대로까지m','길_교차로300m','길_통행중심성']
for f in F: e['z_'+f]=(e[f]-e[f].mean())/e[f].std()
for g in [None,'커피','치킨']:
    x=e if g is None else e[e.업종==g]; print(f'\n## {g or "전체"} n={len(x)} 사건={x.y.sum()}')
    for f in F:
        m=smf.logit(f'y~Q("z_{f}")+C(브랜드)+C(연)',x).fit(disp=0,method='bfgs',maxiter=400); k=f'Q("z_{f}")'
        print(f'  {f}: OR(1SD) {np.exp(m.params[k]):.2f}, p={m.pvalues[k]:.3f}')
e['대로변']=np.where(e.길_접한최고위계>=4,'대로변(왕복4차로급↑)',np.where(e.길_접한최고위계==3,'중로','골목·이면'))
e['기대']=e.groupby('브랜드').y.transform('mean')
for g in ['커피','치킨']:
    t=e[e.업종==g].groupby('대로변').agg(n=('y','size'),폐점률=('y','mean'),기대=('기대','mean')); t['브랜드대비']=(t.폐점률/t.기대).round(2); print(f'\n{g}'); print(t.round(3).to_string())
