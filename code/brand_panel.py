# 대안 점검: 공정위 외식 브랜드-연도 패널로 "다음 해 가맹점 20% 이상 감소" 예측 (기준선: 작년 감소 규칙)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d=pd.read_csv(f'{O}/data/raw/ftc_eatout_2017_2025.csv',dtype=str)
num=['frcsCnt','dmsCnt','prvyrFrcsCnt','prvyrDmsCnt','jngBzmnJngAmt','jngBzmnEduAmt','jngBzmnGrnteAmt','jngBzmnEtcAmt','frstCtrtYycnt','prlgCtrtYycnt','intrrAmt','storCrtrArea','avgSlsAmt','areaUnitAvgSlsAmt','prvyrAvgSlsAmt']
for c in num: d[c]=pd.to_numeric(d[c],errors='coerce')
d['yr']=d.jngBizCrtrYr.astype(int); d=d.drop_duplicates(['alsBrandMngNo','yr'])
nx=d[['alsBrandMngNo','yr','frcsCnt']].assign(yr=lambda x:x.yr-1).rename(columns={'frcsCnt':'next'})
p=d.merge(nx,on=['alsBrandMngNo','yr'],how='inner'); p=p[p.frcsCnt>=10]
p['y']=(p.next<0.8*p.frcsCnt).astype(int)
p['g1']=p.frcsCnt/p.prvyrFrcsCnt.replace(0,np.nan)-1; p['dms비']=p.dmsCnt/(p.dmsCnt+p.frcsCnt); p['업력']=p.yr-pd.to_datetime(p.jngBizStrtYmd,errors='coerce').dt.year
p['창업비']=p[['jngBzmnJngAmt','jngBzmnEduAmt','jngBzmnGrnteAmt','jngBzmnEtcAmt','intrrAmt']].sum(axis=1,min_count=1); p['매출g']=p.avgSlsAmt/p.prvyrAvgSlsAmt-1
F=['g1','dms비','업력','창업비','avgSlsAmt','areaUnitAvgSlsAmt','매출g','frstCtrtYycnt','intrrAmt','frcsCnt']
X=pd.concat([np.sign(p[F])*np.log1p(p[F].abs()),pd.get_dummies(p.tpbizMclsfNm,dtype=float)],axis=1)
tr=p.yr<=2021; te=p.yr>=2022
print('행',len(p),'브랜드',p.alsBrandMngNo.nunique(),'사건률',p.y.mean().round(3),'평가 n',te.sum(),'사건',p.y[te].sum())
base=-p.g1.fillna(0)[te]; print('기준선(작년 증가율 낮을수록 위험) AUC',round(roc_auc_score(p.y[te],base),3))
gb=HGB(max_depth=3,learning_rate=0.05,max_iter=300,random_state=0).fit(X[tr],p.y[tr]); print('GBM AUC',round(roc_auc_score(p.y[te],gb.predict_proba(X[te])[:,1]),3))
X=X.replace([np.inf,-np.inf],np.nan); Xf=X.fillna(X[tr].median()); Z=(Xf-Xf[tr].mean())/Xf[tr].std().replace(0,1)
lr=LogisticRegression(C=0.3,max_iter=3000).fit(Z[tr],p.y[tr]); print('로지스틱 AUC',round(roc_auc_score(p.y[te],lr.predict_proba(Z[te])[:,1]),3))
