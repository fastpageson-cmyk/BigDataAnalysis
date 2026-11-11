# 폐업 예측 모델 비교: 로지스틱(L1/L2/EN/교호작용) · 랜덤포레스트 · ExtraTrees · GBM · 배깅 · 조사별 스태킹 · 소프트보팅
#  입력: indicators.py가 만든 _indicators.pkl (치킨 B_넓음, 2018–22 개업, y = 3년 안 실제 폐업). 전처리는 model.py와 같음
#  분할: 학습 2018–20 개업 / 평가 2021–22 개업 (model.py와 같은 시간 분할).
#   - 하이퍼파라미터·피처 묶음·최종 모델 선택은 학습 기간 반복 5겹 CV(3회) AUC로만 한다. 평가 기간은 마지막에 한 번만 본다.
#   - 보조: 2018–22 전체를 층화 무작위 70/30으로 20번 나눠 상위 모델 순위가 흔들리는지 확인
#  결과 → data/processed/model_compare_cv.csv, model_compare_test.csv, model_compare_repeat.csv
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,BaggingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold,StratifiedShuffleSplit,cross_val_predict,StratifiedKFold
from sklearn.metrics import roc_auc_score,brier_score_loss
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'

# ---- 전처리 (model.py와 동일) ----
d=pd.read_pickle(f'{P}/_indicators.pkl').reset_index(drop=True)
fr=d.프랜차이즈==1; tr=(d.연<=2020).values; te=~tr
d['공정위없음']=(fr&d.공정위_이탈률.isna()).astype(int)
for v in ['공정위_이탈률','공정위_확장률','공정위_명의변경률','공정위_l가맹점','브랜드업력','같은브랜드500','같은브랜드1km','상권같은브랜드']:
    med=d.loc[tr&fr,v].median(); d[v]=np.where(fr,d[v].fillna(med),0)
d['상권치킨수']=d.상권치킨수.fillna(0)
for v in ['동_저녁낮비','동_밤저녁비','동_비중2029','저녁인구증가']: d[v]=d[v].fillna(d.loc[tr,v].median())
for t_ in ['직장가','주거지','먹자골목']: d['유형_'+t_]=(d.사람유형==t_).astype(int)

# ---- 피처 묶음: 지금까지의 조사 단위로 나눔 ----
G={'브랜드':['l브랜드률','프랜차이즈','브랜드업력','공정위_이탈률','공정위_확장률','공정위_명의변경률','공정위_l가맹점','공정위없음'],
   '점포':['면적','지하','2층이상','휴게','다중이용','여름개업'],
   '경쟁·포화':['l경쟁500','상권치킨수','신규개업500_1y','폐업500_1y','폐업비500_1y','프랜비중500','음식점500','음식점폐업비500_1y','치킨비중500','같은브랜드500','같은브랜드1km','상권같은브랜드'],
   '수요·생활인구':['l저녁인구500','거주비','비중3049','저녁인구증가','l수요경쟁비','l역','동_저녁낮비','동_밤저녁비','동_비중2029'],
   '상권유형·자리':['상권밖','유형_직장가','유형_주거지','유형_먹자골목','자리폐업10y']}
ALL=sum(G.values(),[])
S={'S3 브랜드+면적+업력':['l브랜드률','면적','브랜드업력'],
   'S9 채택지표':['l브랜드률','프랜차이즈','면적','브랜드업력','공정위_명의변경률','공정위_이탈률','공정위없음','음식점500','치킨비중500'],
   f'S{len(ALL)} 후보 전부':ALL}
X=d[ALL].astype(float); y=d.y.values
Xtr,ytr,Xte,yte=X[tr],y[tr],X[te],y[te]
print(f'학습 2018–20 {tr.sum()}곳(폐업 {ytr.mean():.3f}) / 평가 2021–22 {te.sum()}곳(폐업 {yte.mean():.3f}), 후보 {len(ALL)}개')

CV=RepeatedStratifiedKFold(n_splits=5,n_repeats=3,random_state=0)
def cv_auc(est,cols):
    s=[]
    for a,b in CV.split(Xtr,ytr):
        m=clone(est).fit(Xtr.iloc[a][cols],ytr[a]); s.append(roc_auc_score(ytr[b],m.predict_proba(Xtr.iloc[b][cols])[:,1]))
    return np.mean(s),np.std(s)/np.sqrt(len(s))

LR=lambda **k: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,**k))
# 후보 모델(이름, 추정기, 하이퍼파라미터 격자) — 격자는 학습 CV로만 고름
MODELS={
 '로지스틱 L2':[LR(C=c) for c in [0.01,0.03,0.1,0.3,1,10]],
 '로지스틱 L1':[LR(C=c,penalty='l1',solver='liblinear') for c in [0.01,0.03,0.1,0.3,1]],
 '로지스틱 ElasticNet':[LR(C=c,penalty='elasticnet',solver='saga',l1_ratio=r) for c in [0.03,0.1,0.3] for r in [0.3,0.7]],
 '로지스틱 2차 교호작용(L1)':[make_pipeline(StandardScaler(),PolynomialFeatures(2,include_bias=False),StandardScaler(),LogisticRegression(C=c,penalty='l1',solver='liblinear',max_iter=5000)) for c in [0.01,0.03,0.1]],
 '랜덤포레스트':[RandomForestClassifier(n_estimators=500,max_depth=md,min_samples_leaf=l,max_features=mf,n_jobs=-1,random_state=0) for md in [4,8,None] for l in [20,50] for mf in ['sqrt',0.5]],
 'ExtraTrees':[ExtraTreesClassifier(n_estimators=500,max_depth=md,min_samples_leaf=l,max_features=mf,n_jobs=-1,random_state=0) for md in [6,None] for l in [20,50] for mf in ['sqrt',0.5]],
 'GBM(부스팅)':[HistGradientBoostingClassifier(max_depth=md,learning_rate=lr,max_iter=it,min_samples_leaf=40,l2_regularization=1.0,random_state=0) for md in [2,3] for lr,it in [(0.03,150),(0.03,300),(0.1,100)]],
 '배깅(로지스틱)':[BaggingClassifier(LR(C=c),n_estimators=200,max_samples=ms,max_features=mf,n_jobs=-1,random_state=0) for c in [0.1,1] for ms in [0.8,1.0] for mf in [0.7,1.0]],
 '배깅(GBM)':[BaggingClassifier(HistGradientBoostingClassifier(max_depth=2,learning_rate=0.05,max_iter=150,min_samples_leaf=40,random_state=0),n_estimators=30,max_samples=0.8,max_features=mf,n_jobs=-1,random_state=0) for mf in [0.7,1.0]],
}
rows=[]; best={}
for sname,cols in S.items():
    for mname,grid in MODELS.items():
        if mname=='로지스틱 2차 교호작용(L1)' and len(cols)>12: continue   # 40개 2차항(820개)은 생략
        res=[(cv_auc(e,cols),e) for e in grid]; (m_,se),e=max(res,key=lambda r:r[0][0])
        rows.append(dict(피처=sname,모델=mname,CV_AUC=m_,CV_SE=se,격자수=len(grid))); best[(sname,mname)]=(e,cols)
        print(f'  {sname:16s} {mname:22s} CV AUC {m_:.4f} ±{se:.4f} (격자 {len(grid)})',flush=True)

# ---- 조사별 스태킹: 묶음마다 로지스틱 → 학습 기간 out-of-fold 예측 → 상위 로지스틱(가중치 = 조사별 기여) ----
class GroupStack:
    def __init__(s,C=0.3): s.C=C
    def get_params(s,deep=False): return {'C':s.C}
    def set_params(s,**k): s.__dict__.update(k); return s
    def fit(s,X,y):
        s.base={g:LR(C=s.C).fit(X[c],y) for g,c in G.items()}
        oof=np.column_stack([cross_val_predict(LR(C=s.C),X[c],y,cv=StratifiedKFold(5,shuffle=True,random_state=1),method='predict_proba')[:,1] for c in G.values()])
        s.meta=LogisticRegression(max_iter=5000).fit(np.log(oof/(1-oof)),y); return s
    def predict_proba(s,X):
        z=np.column_stack([s.base[g].predict_proba(X[c])[:,1] for g,c in G.items()]); p=s.meta.predict_proba(np.log(z/(1-z)))[:,1]; return np.column_stack([1-p,p])
res=[(cv_auc(GroupStack(c),ALL),GroupStack(c)) for c in [0.03,0.3]]; (m_,se),e=max(res,key=lambda r:r[0][0])
rows.append(dict(피처=f'S{len(ALL)} 후보 전부',모델='조사별 스태킹',CV_AUC=m_,CV_SE=se,격자수=2)); best[(f'S{len(ALL)} 후보 전부','조사별 스태킹')]=(e,ALL)
print(f'  조사별 스태킹 CV AUC {m_:.4f} ±{se:.4f}')

R=pd.DataFrame(rows).sort_values('CV_AUC',ascending=False).reset_index(drop=True)
# ---- 소프트보팅: CV 상위 서로 다른 모델 3개 평균 ----
class Vote:
    def __init__(s,parts): s.parts=parts
    def get_params(s,deep=False): return {'parts':s.parts}
    def fit(s,X,y): s.f=[(clone(e).fit(X[c],y),c) for e,c in s.parts]; return s
    def predict_proba(s,X): p=np.mean([m.predict_proba(X[c])[:,1] for m,c in s.f],0); return np.column_stack([1-p,p])
top=R.drop_duplicates('모델').head(3); parts=[best[(a,b)] for a,b in zip(top.피처,top.모델)]
vm,vs=cv_auc(Vote(parts),ALL); best[('상위3 혼합','소프트보팅')]=(Vote(parts),ALL)
R=pd.concat([R,pd.DataFrame([dict(피처='상위3 혼합',모델='소프트보팅('+'+'.join(top.모델)+')',CV_AUC=vm,CV_SE=vs,격자수=1)])]).sort_values('CV_AUC',ascending=False).reset_index(drop=True)
R.loc[R.모델.str.startswith('소프트보팅'),'모델']='소프트보팅'; R['_full']=R.모델
pd.set_option('display.width',250)
print('\n[학습 기간 반복 5겹 CV AUC 순위]'); print(R.drop(columns='_full').round(4).head(20).to_string())
R.drop(columns='_full').to_csv(f'{P}/model_compare_cv.csv',encoding='utf-8-sig',index=False)

# ---- 평가 기간 (한 번): 기준선 = 현행 ★ 3변수 로지스틱(model.py) ----
ref=LR(C=1e6).fit(Xtr[S['S3 브랜드+면적+업력']],ytr).predict_proba(Xte[S['S3 브랜드+면적+업력']])[:,1]
brand=d.loc[te,'브랜드률'].values
rng=np.random.default_rng(0); idx=[rng.integers(0,len(yte),len(yte)) for _ in range(2000)]
def diff(a,b): v=np.array([roc_auc_score(yte[i],a[i])-roc_auc_score(yte[i],b[i]) for i in idx]); return np.percentile(v,[2.5,97.5])
T=[]
def ev(name,p):
    lo,hi=diff(p,ref); q80,q20=np.quantile(p,.8),np.quantile(p,.2)
    T.append(dict(모델=name,평가_AUC=roc_auc_score(yte,p),AUC차_vs현행=roc_auc_score(yte,p)-roc_auc_score(yte,ref),CI하=lo,CI상=hi,Brier=brier_score_loss(yte,p),
                  상위20_폐업률=yte[p>=q80].mean(),하위20_폐업률=yte[p<=q20].mean()))
ev('기준선 ③ 브랜드 과거 폐업률',brand); ev('현행 ★ 3변수 로지스틱',ref)
for _,r in R.head(12).iterrows():
    key=('상위3 혼합','소프트보팅') if r.모델=='소프트보팅' else (r.피처,r.모델); e,cols=best[key]
    ev(f'{r.모델} [{r.피처}] (CV {r.CV_AUC:.3f})',clone(e).fit(Xtr[cols],ytr).predict_proba(Xte[cols])[:,1] if not isinstance(e,(Vote,GroupStack)) else e.fit(Xtr,ytr).predict_proba(Xte)[:,1])
TT=pd.DataFrame(T); print(f'\n[평가 2021–22 개업 {te.sum()}곳 — CV 상위 12개만, 한 번 평가]'); print(TT.round(4).to_string())
TT.to_csv(f'{P}/model_compare_test.csv',encoding='utf-8-sig',index=False)

# 스태킹 상위 가중치(조사별 기여) 보고
gs=GroupStack(best[(f'S{len(ALL)} 후보 전부','조사별 스태킹')][0].C).fit(Xtr,ytr)
print('\n[조사별 스태킹 상위 로지스틱 가중치 — 묶음 예측 로짓 1단위당]'); print(pd.Series(gs.meta.coef_[0],index=list(G)).round(3).to_string())
for g,c in G.items(): print(f'  {g:12s} 단독 평가 AUC {roc_auc_score(yte,gs.base[g].predict_proba(Xte[c])[:,1]):.3f}')

# ---- 보조: 무작위 70/30 × 20번 (상위 모델 + 현행) ----
sel=[('S3 브랜드+면적+업력','로지스틱 L2')]+[(a,b) for a,b in zip(R.피처,R.모델) if b!='소프트보팅'][:5]
sel=list(dict.fromkeys(sel)); RR=[]
for k,(a,b) in enumerate(sel):
    e,cols=best[(a,b)]; s=[]
    for i,(A,B) in enumerate(StratifiedShuffleSplit(20,test_size=0.3,random_state=7).split(X,y)):
        m=(e.__class__(e.C) if isinstance(e,GroupStack) else clone(e)).fit(X.iloc[A][cols] if not isinstance(e,GroupStack) else X.iloc[A],y[A])
        s.append(roc_auc_score(y[B],m.predict_proba(X.iloc[B][cols] if not isinstance(e,GroupStack) else X.iloc[B])[:,1]))
    RR.append(dict(모델=f'{b} [{a}]',평균AUC=np.mean(s),SD=np.std(s),최소=np.min(s),최대=np.max(s),_s=s))
base=np.array(RR[0]['_s'])
for r in RR: r['현행보다_높은_비율']=(np.array(r['_s'])>base).mean(); del r['_s']
RR=pd.DataFrame(RR); print('\n[보조: 2018–22 층화 무작위 70/30 × 20번]'); print(RR.round(4).to_string())
RR.to_csv(f'{P}/model_compare_repeat.csv',encoding='utf-8-sig',index=False)
