# h=3 실제폐점: 변수 묶음별 추가 효과(부트스트랩 CI), 브랜드 고정효과 로지스틱 계수, 상권 시점일치 부분표본
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from eval_h import s,lab,pairauc,prep,STORE,COMP,DL,DM,END
H=3; Y='진짜폐점'
last=END-pd.DateOffset(years=H); te_lo=last-pd.DateOffset(years=2); tr_lo=te_lo-pd.DateOffset(years=6)
d=s[(s.open>tr_lo)&(s.open<=last)].copy(); y=lab(d,H,Y).values; tr=(d.open<=te_lo).values; te=~tr
br=pd.Series(y[tr]).groupby(d.브랜드.values[tr]).agg(['sum','count']); pr=y[tr].mean(); d['브랜드률']=d.브랜드.map((br['sum']+20*pr)/(br['count']+20)).fillna(pr)
yt=y[te]; g=d.브랜드.values[te]; k=int(te.sum()*.1)
P={'기준선②':d.브랜드률.values[te]+1e-9*np.random.RandomState(0).rand(te.sum())}
PW={}
for nm,F in [('+점포·경쟁',STORE+COMP),('+동 인허가',STORE+COMP+DL),('+동 상권(매출·인구)',STORE+COMP+DL+DM)]:
    Z=prep(d,tr,['브랜드률','업종']+F); P[nm]=LogisticRegression(C=0.1,max_iter=3000).fit(Z[tr],y[tr]).predict_proba(Z[te])[:,1]
    Zb=prep(d,tr,['브랜드']+F); PW[nm]=LogisticRegression(C=0.1,max_iter=3000).fit(Zb[tr],y[tr]).predict_proba(Zb[te])[:,1]
rs=np.random.RandomState(1); B=[rs.randint(0,len(yt),len(yt)) for _ in range(500)]
def ci(f): v=np.array([f(ix) for ix in B]); return f'{np.median(v):.3f} [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]'
top=lambda q,ix: yt[ix][np.argsort(-q[ix])[:k]].mean()
print(f'평가 n={te.sum()} 사건={yt.sum()} 상위10%={k}곳')
for nm,q in P.items():
    print(f'{nm:16s} AUC {roc_auc_score(yt,q):.3f} | top10정밀 {top(q,np.arange(len(yt))):.3f} | top10 차(−기준선) {ci(lambda ix: top(q,ix)-top(P["기준선②"],ix)) if nm!="기준선②" else "-"}'+(f' | 브랜드내부쌍AUC {ci(lambda ix: pairauc(yt[ix],PW[nm][ix],g[ix]))}' if nm in PW else ''))
# 브랜드 고정효과 로지스틱(2012~2023-11 개업 전체, 추론용)
e=s[(s.open.dt.year>=2012)&(s.open<=last)].copy(); ye=lab(e,H,Y)
F=STORE+COMP+DL+DM+['상권시점일치']
X=e[F].astype(float); X=X.fillna(X.median()); X=(X-X.mean())/X.std()
X=pd.concat([X,pd.get_dummies(e.브랜드,drop_first=True,dtype=float),pd.get_dummies(e.연,prefix='연',drop_first=True,dtype=float)],axis=1)
m=sm.Logit(ye.values,sm.add_constant(X)).fit(disp=0,maxiter=200,method='lbfgs')
c=pd.DataFrame({'계수(1SD)':m.params,'p':m.pvalues}).loc[F].round(3); c['OR']=np.exp(c['계수(1SD)']).round(2)
print(f'\n## 브랜드·연도 고정효과 로지스틱 n={len(e)} 사건={ye.sum()}'); print(c.sort_values('p').to_string())
# 상권 시점일치 부분표본: 2021Q2~2023-11 개업, 브랜드 층화 5겹 CV
f=s[(s.상권시점일치==1)&(s.open<=last)].copy(); yf=lab(f,H,Y).values; print(f'\n## 상권 시점일치 부분표본 n={len(f)} 사건={yf.sum()}')
for nm,FF in [('점포·경쟁+동인허가',STORE+COMP+DL),('+동 상권',STORE+COMP+DL+DM)]:
    pw=np.zeros(len(f))
    for a,b in StratifiedKFold(5,shuffle=True,random_state=0).split(f,yf):
        m_=np.zeros(len(f),bool); m_[a]=True; Z=prep(f,m_,['브랜드']+FF); pw[b]=LogisticRegression(C=0.1,max_iter=3000).fit(Z[m_],yf[m_]).predict_proba(Z[~m_])[:,1]
    print(nm,'브랜드내부쌍AUC',round(pairauc(yf,pw,f.브랜드.values),3),'전체AUC',round(roc_auc_score(yf,pw),3))
