# 상권 정의 비교(5년 실제폐점, 시간 분리): 기준선 / 나이브 5유형 프로필 / 데이터 기반 주변 구성 / 둘 다
#  각 정의마다 업종×피처 상호작용(궁합)을 넣은 로지스틱. 지표: AUC, 상위10% 정밀도, 같은 상권·업종 브랜드 쌍 정확도, 브랜드 내부 쌍 AUC
import os,io,contextlib,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
with contextlib.redirect_stdout(io.StringIO()): from rec_eval import pairs,PROF,COMP,DL,ST
from eval_h import pairauc
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); H=5
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/area_profile.pkl'),on='상권코드').join(pd.read_pickle(f'{O}/data/processed/micro_feats.pkl'))
MIC=[c for c in s.columns if c.startswith(('주변_','현재_'))]
last=END-pd.DateOffset(years=H); te_lo=last-pd.DateOffset(years=3)
d=s[(s.open.dt.year>=2008)&(s.open<=last)].copy(); y=((d.close<=d.open+pd.DateOffset(years=H))&d.진짜폐점).astype(int); tr=d.open<=te_lo; te=~tr
br=y[tr].groupby(d.브랜드[tr]).agg(['sum','count']); pr=y[tr].mean(); d['브랜드률']=d.브랜드.map((br['sum']+20*pr)/(br['count']+20)).fillna(pr)
d['상권코드']=d.상권코드.fillna('없음')
def Zm(F,inter):
    X=pd.concat([d[['브랜드률']+F],pd.get_dummies(d.업종,dtype=float)],axis=1); X=X.fillna(X[tr].median()); Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    for g in d.업종.unique():
        gg=(d.업종==g).astype(float)
        for c in inter: Z[f'{g}×{c}']=gg*Z[c]
    return Z
yt=y[te].values; k=int(te.sum()*.1); base=d.브랜드률+1e-9*np.random.RandomState(0).rand(len(d))
def report(nm,p):
    p=pd.Series(p,index=d.index); pt=p[te].values
    print(f'{nm:24s} AUC {roc_auc_score(yt,pt):.3f} | 상위10% {yt[np.argsort(-pt)[:k]].mean():.3f} | 같은상권·업종 브랜드쌍 {pairs(d[te],p,y)[0]:.3f} | 브랜드내부쌍AUC {pairauc(yt,pt,d.브랜드.values[te]):.3f}')
print(f'평가: {te_lo:%Y-%m}~{last:%Y-%m} 개업 n={te.sum()} 사건={yt.sum()}')
report('기준선(브랜드 평균)',base)
for nm,F,I in [('나이브 5유형 프로필',ST+COMP+DL+PROF,PROF),('데이터기반 주변구성',ST+COMP+DL+MIC,MIC),('둘 다',ST+COMP+DL+PROF+MIC,PROF+MIC)]:
    Z=Zm(F,I); m=LogisticRegression(C=0.05,max_iter=5000).fit(Z[tr],y[tr]); report(nm,m.predict_proba(Z)[:,1])

# 브랜드×피처 궁합(상위 15개 브랜드) 버전, 상권 안 점포만(rec_eval과 같은 조건)
print('\n-- 브랜드×피처 궁합, 상권 안 점포만 --')
inside=(d.상권코드!='없음')&d['외식매출(log)'].notna()
TOP=d.브랜드[tr].value_counts().index[:15]
def Zb(F,inter):
    X=pd.concat([d[['브랜드률']+F],pd.get_dummies(d.업종,dtype=float)],axis=1); X=X.fillna(X[tr].median()); Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    for b in TOP:
        bb=(d.브랜드==b).astype(float)
        for c in inter: Z[f'{b}×{c}']=bb*Z[c]
    return Z
d0,y0,tr0,te0=d,y,tr,te
d=d0[inside]; y=y0[inside]; tr=tr0[inside]; te=te0[inside]; yt=y[te].values; k=int(te.sum()*.1); base=d.브랜드률+1e-9*np.random.RandomState(0).rand(len(d))
print(f'n평가={te.sum()} 사건={yt.sum()}'); report('기준선(브랜드 평균)',base)
for nm,F,I in [('나이브 5유형 × 브랜드',ST+COMP+DL+PROF,PROF),('데이터기반 × 브랜드',ST+COMP+DL+MIC,MIC)]:
    Z=Zb(F,I); m=LogisticRegression(C=0.05,max_iter=5000).fit(Z[tr],y[tr]); report(nm,m.predict_proba(Z)[:,1])
