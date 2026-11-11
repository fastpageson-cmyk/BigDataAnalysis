# 폐업 예측 피처 영향도 점검 (치킨, chicken_wide.py B_넓음, y = 개업 3년 안 실제 폐업)
#  현행 모델: chicken_wide.py  y ~ log1p(500m 밀도) + 프랜차이즈 + 면적 (+C(연))
#  후보 피처: 개업 시점에 알 수 있는 인허가 값 + 위치 계산값. 학습 2015–19 개업 → 평가 2020–22 개업 (시간 분할)
#  영향도 3가지: ① 피처 하나만 쓴 평가 AUC ② 전체 모델에서 뺐을 때 AUC 감소(drop-column) ③ 순열 중요도
#  누수 점검: 종사자수·영업장주변구분은 사후 수정될 수 있어 '결측 여부'가 y와 엮이는지 따로 출력
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(cw)
c=cw.STORES['B_넓음'].copy(); d=cw.D['B_넓음'].copy(); d['y']=d.y_실제

# ---- 후보 피처 (모두 개업 시점 기준) ----
T=cKDTree(c[['X','Y']].values); co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values
hist,same=[],[]
for r in d.itertuples():
    t=np.datetime64(r.open)
    q=np.array(T.query_ball_point([r.X,r.Y],10),dtype=int); hist.append(((cc[q]<t)).sum())                 # 10m 안 과거 치킨집 폐업 횟수
    if r.프랜차이즈:
        q=np.array(T.query_ball_point([r.X,r.Y],1000),dtype=int); same.append(((c.브랜드.values[q]==r.브랜드)&(co[q]<t)&(cc[q]>t)).sum())
    else: same.append(0)
d['자리이력']=hist; d['같은브랜드1km']=same
d['종사자']=pd.to_numeric(d.남성종사자수,errors='coerce')+pd.to_numeric(d.여성종사자수,errors='coerce')
addr=d.지번주소.fillna('')
d['지하']=addr.str.contains('지하|B1|b1').astype(int); d['2층이상']=addr.str.contains(r'[2-9]층|1\d층').astype(int)
d['구']=addr.str.extract(r'서울특별시\s+(\S+구)')[0].fillna('기타')
d['휴게']=(d.src=='cafe').astype(int); d['다중이용']=(d.다중이용업소여부=='Y').astype(int)
d['개업월']=d.open.dt.month; d['로그밀도']=np.log1p(d.밀도)

tr=d.연<=2019; te=~tr
# 범주형은 학습 기간 폐업률로 인코딩(수축 k=20)해 평가 기간에 그대로 적용 → 누수 없음
def enc(col,k=20):
    s=d[col] if isinstance(col,str) else col
    g=d[tr].y.groupby(s[tr]).agg(['sum','size']); m=d[tr].y.mean(); r=(g['sum']+k*m)/(g['size']+k); return s.map(r).fillna(m)
d['브랜드_률']=enc(d.브랜드.fillna('개인')); d['업태_률']=enc('업태구분명'); d['구_률']=enc('구'); d['주변_률']=enc(d.영업장주변구분명.fillna('결측'))

FEAT={'프랜차이즈':'프랜차이즈','브랜드(학습기간 폐업률)':'브랜드_률','면적':'면적','500m 치킨집 밀도':'로그밀도','개업연도':'연',
      '일반/휴게':'휴게','업태':'업태_률','자치구':'구_률','지하':'지하','2층 이상':'2층이상','자리 이력(10m 과거 폐업)':'자리이력',
      '같은 브랜드 1km':'같은브랜드1km','종사자수':'종사자','영업장주변구분':'주변_률','다중이용업소':'다중이용','개업월':'개업월'}
X=d[list(FEAT.values())].astype(float); y=d.y
print(f'표본 {len(d)} (학습 {tr.sum()} / 평가 {te.sum()}), 3년 실제 폐업률 학습 {y[tr].mean():.3f} / 평가 {y[te].mean():.3f}')

def fit(cols):
    m=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,min_samples_leaf=40,random_state=0).fit(X.loc[tr,cols],y[tr])
    return m,roc_auc_score(y[te],m.predict_proba(X.loc[te,cols])[:,1])
cols=list(FEAT.values()); full,auc_full=fit(cols)
_,auc_cur=fit(['로그밀도','프랜차이즈','면적'])
print(f'\n평가 AUC: 현행 피처(밀도·프랜차이즈·면적) {auc_cur:.3f} / 후보 전체 {auc_full:.3f} / 브랜드율 하나만 {roc_auc_score(y[te],X.loc[te,"브랜드_률"]):.3f}')

pi=permutation_importance(full,X.loc[te,cols],y[te],scoring='roc_auc',n_repeats=30,random_state=0)
rows=[]
for nm,cn in FEAT.items():
    s=X.loc[te,cn]; a1=roc_auc_score(y[te],s.fillna(s.median())) if s.nunique()>1 else .5
    _,ad=fit([x for x in cols if x!=cn]); i=cols.index(cn)
    rows.append(dict(피처=nm,현행사용='●' if cn in ('로그밀도','프랜차이즈','면적','연') else '',결측률=X[cn].isna().mean(),
                     단독AUC=max(a1,1-a1),방향='+' if a1>=.5 else '−',뺐을때_AUC감소=auc_full-ad,순열중요도=pi.importances_mean[i],순열sd=pi.importances_std[i]))
r=pd.DataFrame(rows).sort_values('순열중요도',ascending=False)
pd.set_option('display.width',250); print('\n[피처 영향도] 방향 +: 값이 클수록 폐업↑'); print(r.round(4).to_string(index=False))

print('\n[누수 점검] 사후 수정 가능 컬럼의 결측 여부별 폐업률')
for cn in ['종사자','주변_률']:
    src='종사자' if cn=='종사자' else None
    miss=d['종사자'].isna() if cn=='종사자' else d.영업장주변구분명.isna()
    print(f'  {cn}: 결측 {d[miss].y.mean():.3f} (n={miss.sum()}) vs 있음 {d[~miss].y.mean():.3f} (n={(~miss).sum()})')
r.to_csv(f'{O}/data/processed/feature_importance.csv',index=False,encoding='utf-8-sig')
