# 망한 점포 주변의 공통점: ① 피처별 브랜드·연도 고정효과 로지스틱(5년 실제폐점) ② 업종별 차이 ③ 망한 점포 주변 유형(군집) → 전체 점포에 적용해 폐점률 비교
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans
from statsmodels.stats.multitest import multipletests
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); H=5
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/micro_feats.pkl'))
F=[c for c in s.columns if c.startswith(('주변_','현재_'))]+['최근접같은브랜드m','같은브랜드500']
e=s[(s.open.dt.year>=2008)&(s.open<=END-pd.DateOffset(years=H))].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=H))&e.진짜폐점).astype(int)
Z=e[F].fillna(e[F].median()); Z=(Z-Z.mean())/Z.std(); e[[f'z{i}' for i in range(len(F))]]=Z.values
print(f'n={len(e)} 실제폐점={e.y.sum()} ({e.y.mean():.3f})')
def fe(df,label):
    R=[]
    for i,f in enumerate(F):
        m=smf.logit(f'y~z{i}+C(브랜드)+C(연)',df).fit(disp=0,method='bfgs',maxiter=400); R.append((f,np.exp(m.params[f'z{i}']),m.pvalues[f'z{i}']))
    r=pd.DataFrame(R,columns=['피처','OR(1SD)','p']); r['q(FDR)']=multipletests(r.p,method='fdr_bh')[1]; r=r.sort_values('p')
    print(f'\n## {label} (n={len(df)}, 사건={df.y.sum()})'); print(r[r.p<0.1].round(3).to_string(index=False)); return r
allr=fe(e,'전체 브랜드: 브랜드·연도 고정효과, 피처 하나씩')
for g in ['커피','치킨']: fe(e[e.업종==g],f'{g} 브랜드만')
# ③ 망한 점포 주변 유형
fail=Z[e.y==1]; km=KMeans(5,n_init=30,random_state=0).fit(fail); e['주변유형']=km.predict(Z)
cen=pd.DataFrame(km.cluster_centers_,columns=F)
e['기대']=e.groupby('브랜드').y.transform('mean')
t=e.groupby('주변유형').agg(점포=('y','size'),폐점=('y','sum'),폐점률=('y','mean'),브랜드기대=('기대','mean')); t['브랜드대비']=t.폐점률/t.브랜드기대
for k in cen.index: t.loc[k,'특징(+)']=', '.join(cen.loc[k].sort_values(ascending=False).index[:4].str.replace('주변_|현재_|비중','',regex=True)); t.loc[k,'특징(−)']=', '.join(cen.loc[k].sort_values().index[:3].str.replace('주변_|현재_|비중','',regex=True))
print('\n## 망한 점포 주변 유형 5개 → 전체 점포 적용'); print(t.round(3).to_string())
e[['관리번호','주변유형']].to_pickle(f'{O}/data/processed/_fail_types.pkl'); allr.to_csv(f'{O}/data/processed/fail_features.csv',index=False,encoding='utf-8-sig')
