# 보조지표를 겹친 폐업 예측 모델 (치킨 B_넓음, 2018–22 개업, y = 3년 안 실제 폐업)
#  입력: indicators.py가 만든 _indicators.pkl (개업 시점 값만)
#  분할: 학습 2018–20 개업 / 평가 2021–22 개업 (시간 순서, 미래 정보 차단). 지표 선택도 학습 기간에서만 한다.
#  ① L1 로지스틱(교차검증)으로 학습 기간에서 지표 고르기 → ② 고른 지표로 일반 로지스틱 = 방정식
#  기준선: 전원 생존 / 전체 평균 / 프랜차이즈 평균 / 브랜드 과거 폐업률 / 브랜드+면적. 참고: 후보 전부 GBM
#  결과 → data/processed/model_equation.csv, model_eval.csv
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,brier_score_loss,confusion_matrix
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
d=pd.read_pickle(f'{P}/_indicators.pkl').reset_index(drop=True)
fr=d.프랜차이즈==1
# 프랜차이즈 전용 지표: 개인은 0, 공정위에 없는 프랜차이즈는 학습기간 프랜차이즈 중앙값으로 채우고 '공정위없음' 표시
tr=d.연<=2020; te=~tr
d['공정위없음']=(fr&d.공정위_이탈률.isna()).astype(int)
for v in ['공정위_이탈률','공정위_확장률','공정위_명의변경률','공정위_l가맹점','브랜드업력','같은브랜드500','같은브랜드1km','상권같은브랜드']:
    med=d.loc[tr&fr,v].median(); d[v]=np.where(fr,d[v].fillna(med),0)
d['상권치킨수']=d.상권치킨수.fillna(0)
for v in ['동_저녁낮비','동_밤저녁비','동_비중2029','저녁인구증가']: d[v]=d[v].fillna(d.loc[tr,v].median())
for t_ in ['직장가','주거지','먹자골목']: d['유형_'+t_]=(d.사람유형==t_).astype(int)
CAND=['l브랜드률','프랜차이즈','면적','브랜드업력','공정위_이탈률','공정위_확장률','공정위_명의변경률','공정위_l가맹점','공정위없음',
      '같은브랜드500','같은브랜드1km','상권같은브랜드','l경쟁500','상권치킨수','신규개업500_1y','폐업500_1y','폐업비500_1y','프랜비중500',
      '음식점500','음식점폐업비500_1y','치킨비중500','l저녁인구500','거주비','비중3049','저녁인구증가','l수요경쟁비','l역',
      '동_저녁낮비','동_밤저녁비','동_비중2029','상권밖','유형_직장가','유형_주거지','유형_먹자골목','지하','2층이상','휴게','자리폐업10y','여름개업','다중이용']
X=d[CAND].astype(float); y=d.y.values; mu,sd=X[tr].mean(),X[tr].std().replace(0,1); Z=(X-mu)/sd
print(f'학습 2018–20 개업 {tr.sum()}곳(폐업률 {y[tr].mean():.3f}) / 평가 2021–22 개업 {te.sum()}곳(폐업률 {y[te].mean():.3f}), 후보 지표 {len(CAND)}개')

# ① 학습 기간 L1 선택 (5겹, AUC 기준, 1-SE보다 단순한 쪽: 최고 AUC의 0.005 안에서 가장 강한 규제)
cv=LogisticRegressionCV(Cs=np.logspace(-3,0,30),penalty='l1',solver='liblinear',cv=5,scoring='roc_auc',max_iter=5000,random_state=0).fit(Z[tr],y[tr])
sc=cv.scores_[1].mean(0); ok=np.where(sc>=sc.max()-0.005)[0]; C=cv.Cs_[ok.min()]
from sklearn.linear_model import LogisticRegression
l1=LogisticRegression(penalty='l1',solver='liblinear',C=C,max_iter=5000).fit(Z[tr],y[tr])
SEL=[v for v,b in zip(CAND,l1.coef_[0]) if abs(b)>1e-6]
print(f'\n① L1 선택 (C={C:.3f}, 교차검증 AUC {sc[ok.min()]:.3f}): {len(SEL)}개 → {SEL}')

# ② 고른 지표로 일반 로지스틱(원래 단위) = 방정식
m=sm.Logit(y[tr],sm.add_constant(X.loc[tr,SEL])).fit(disp=0)
eq=pd.DataFrame({'계수':m.params,'OR':np.exp(m.params),'p':m.pvalues,'OR_1SD':np.exp(m.params*pd.concat([pd.Series({'const':0}),sd[SEL]]))})
print('\n② 폐업 예측 방정식 (학습 2018–20):  logit(p) = b0 + Σ b·x,  p = 1/(1+e^-logit)'); print(eq.round(4).to_string())
terms=' '.join(f'{"+" if b>=0 else "−"} {abs(b):.3f}·{v}' for v,b in m.params.drop('const').items())
print(f'\n  logit(p) = {m.params["const"]:.3f} {terms}')

# 기준선·비교 모델
P_={}
P_['⓪ 전원 생존(0)']=np.zeros(te.sum())
P_['① 전체 평균']=np.full(te.sum(),y[tr].mean())
P_['② 프랜차이즈 평균']=d.loc[te,'프랜차이즈'].map(d[tr].groupby('프랜차이즈').y.mean()).values
P_['③ 브랜드 과거 폐업률']=d.loc[te,'브랜드률'].values
b2=sm.Logit(y[tr],sm.add_constant(X.loc[tr,['l브랜드률','면적']])).fit(disp=0); P_['④ 브랜드+면적']=b2.predict(sm.add_constant(X.loc[te,['l브랜드률','면적']])).values
P_['★ 보조지표 방정식']=m.predict(sm.add_constant(X.loc[te,SEL])).values
gb=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,min_samples_leaf=40,random_state=0).fit(X[tr],y[tr]); P_['(참고) 후보 전부 GBM']=gb.predict_proba(X[te])[:,1]
yt=y[te]; thr_q=1-y[tr].mean()   # 고위험 판정: 학습기간 폐업률만큼(상위 28%)을 '폐업 위험'으로 찍음
rng=np.random.default_rng(0); idx=[rng.integers(0,len(yt),len(yt)) for _ in range(2000)]
def ci(a,b):
    v=np.array([roc_auc_score(yt[i],a[i])-roc_auc_score(yt[i],b[i]) for i in idx]); return np.percentile(v,[2.5,97.5]),(v<=0).mean()
rows=[]
for k,p in P_.items():
    r=dict(모델=k,AUC=roc_auc_score(yt,p) if np.ptp(p)>0 else .5,Brier=brier_score_loss(yt,p))
    if np.ptp(p)>0:
        cut=np.quantile(p,thr_q); hi=p>=cut; tn,fp,fn,tp=confusion_matrix(yt,hi).ravel()
        r.update(상위20pct_폐업률=yt[p>=np.quantile(p,.8)].mean(),하위20pct_폐업률=yt[p<=np.quantile(p,.2)].mean(),상위20pct_포착=yt[p>=np.quantile(p,.8)].sum()/yt.sum(),
                 정확도=(tp+tn)/len(yt),재현율=tp/(tp+fn),정밀도=tp/(tp+fp))
        if k not in('③ 브랜드 과거 폐업률',): (lo,hi_),pv=ci(p,P_['③ 브랜드 과거 폐업률']); r.update(AUC차_vs브랜드=r['AUC']-roc_auc_score(yt,P_['③ 브랜드 과거 폐업률']),CI하=lo,CI상=hi_)
    else: r.update(정확도=(yt==0).mean(),재현율=0.)
    rows.append(r)
E=pd.DataFrame(rows).set_index('모델'); pd.set_option('display.width',300)
print(f'\n[평가 2021–22 개업 {len(yt)}곳, 실제 폐업 {yt.sum()}곳] 고위험 판정 = 점수 상위 {1-thr_q:.0%}'); print(E.round(3).to_string())
p=P_['★ 보조지표 방정식']; dec=pd.qcut(p,5,labels=['1(안전)','2','3','4','5(위험)'])
print('\n[★ 5분위 보정] 예측 평균 vs 실제 폐업률'); print(pd.DataFrame({'예측':pd.Series(p).groupby(dec).mean(),'실제':pd.Series(yt).groupby(dec).mean(),'n':pd.Series(yt).groupby(dec).size()}).round(3).to_string())
for nm,mask in [('프랜차이즈',d.loc[te,'프랜차이즈'].values==1),('개인',d.loc[te,'프랜차이즈'].values==0)]:
    print(f'  {nm} 안 AUC: 브랜드율 {roc_auc_score(yt[mask],P_["③ 브랜드 과거 폐업률"][mask]):.3f} / ★ {roc_auc_score(yt[mask],p[mask]):.3f}')
eq.to_csv(f'{P}/model_equation.csv',encoding='utf-8-sig'); E.to_csv(f'{P}/model_eval.csv',encoding='utf-8-sig')
