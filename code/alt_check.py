# 대안 방향 점검 (인수인계 '입지 말고 다른 방향' 후보)
#  B. 동 단위: 올해 동 치킨 시장 상태 → 다음 해 동 폐업률·점포당 매출 변화 (dong_chicken_panel.csv, 학습 2021–23 → 평가 2024년 값으로 2025 예측)
#  C. 점포 특성 모델이 기준선 ②(브랜드 평균 폐업률)를 넘는가: feature_importance.py 표본·피처 재사용, 평가 AUC 차이 부트스트랩
#  D. 네이버 리뷰·평점 파일로 폐업 y를 만들 수 있나: 리뷰 점포를 인허가 치킨 표본과 좌표·상호로 연결해 폐업 점포가 몇 곳인지
import os,io,re,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
pd.set_option('display.width',250)

# ---------------- B ----------------
p=pd.read_csv(f'{P}/dong_chicken_panel.csv').sort_values(['동','연'])
p['폐업률']=p.폐업/p.점포; p['개업률']=p.개업/p.점포
p['다음해매출변화']=p.groupby('동').l매출.shift(-1)-p.l매출
p['동평균폐업률']=np.nan
X=['폐업률','개업률','l매출','l점포','프랜비중','저녁비중','d매출','d점포']
print('[B] 동 단위 다음 해 예측 (학습: 2021–23년 상태→다음 해, 평가: 2024년 상태→2025)')
for yv in ['다음해폐업률','다음해매출변화']:
    q=p.dropna(subset=[yv,'l매출','저녁비중']).copy(); q=q[q.다음해폐업률<=1]
    tr=q[q.연<=2023]; te=q[q.연==2024].copy()
    trm=tr.dropna(subset=X); w=te.점포
    def score(pred): ok=pred.notna(); return dict(가중MAE=np.average((te[yv]-pred)[ok].abs(),weights=w[ok]),순위상관=spearmanr(te[yv][ok],pred[ok])[0],n=int(ok.sum()))
    base={'전체 평균(학습)':pd.Series(np.average(tr[yv],weights=tr.점포),index=te.index),
          '동 학습기간 평균':te.동.map(tr.groupby('동').apply(lambda g:np.average(g[yv],weights=g.점포))),
          '올해 값(전년 유지)':te['폐업률'] if yv=='다음해폐업률' else te['d매출']}
    te_=te.copy(); te_[X]=te_[X].fillna(trm[X].median())
    lin=LinearRegression().fit(trm[X],trm[yv],sample_weight=trm.점포)
    gb=HistGradientBoostingRegressor(max_depth=3,max_iter=200,learning_rate=0.05,min_samples_leaf=30,random_state=0).fit(trm[X],trm[yv],sample_weight=trm.점포)
    mods={'선형(8변수)':pd.Series(lin.predict(te_[X]),index=te.index),'GBM(8변수)':pd.Series(gb.predict(te_[X]),index=te.index)}
    r=pd.DataFrame({k:score(v) for k,v in {**base,**mods}.items()}).T
    print(f'\n  y = {yv}  (평가 동 {len(te)}, 평균 {np.average(te[yv],weights=w):.3f})'); print(r.round(4).to_string())
    if yv=='다음해폐업률':
        print('  선형 계수:',{k:round(v,3) for k,v in zip(X,lin.coef_)})
        # 분류로 보기: 다음 해 폐업률 상위 25% 동을 골라내나
        cut=te[yv].quantile(.75); hi=(te[yv]>=cut).astype(int)
        print('  상위25% 동 가려내기 AUC:',{k:round(roc_auc_score(hi[v.notna()],v.dropna()),3) for k,v in {**base,**mods}.items() if v.nunique()>1})

# ---------------- C ----------------
spec=importlib.util.spec_from_file_location('fi',f'{O}/code/feature_importance.py'); fi=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(fi)
d,tr,te,y,Xc=fi.d,fi.tr,fi.te,fi.y,fi.X
cols=list(fi.FEAT.values())
pb=Xc.loc[te,'브랜드_률'].values
pg=fi.full.predict_proba(Xc.loc[te,cols])[:,1]
sc=(Xc-Xc[tr].mean())/Xc[tr].std()
lr=LogisticRegression(max_iter=2000).fit(sc.loc[tr,['브랜드_률','면적']],y[tr]); pl=lr.predict_proba(sc.loc[te,['브랜드_률','면적']])[:,1]
pp=d.loc[te,'프랜차이즈'].map(d[tr].groupby('프랜차이즈').y.mean()).values
yt=y[te].values; rng=np.random.default_rng(0)
def boot(a,b,B=2000):
    diffs=[]
    for _ in range(B):
        i=rng.integers(0,len(yt),len(yt))
        if yt[i].min()==yt[i].max(): continue
        diffs.append(roc_auc_score(yt[i],a[i])-roc_auc_score(yt[i],b[i]))
    diffs=np.array(diffs); return diffs.mean(),np.percentile(diffs,2.5),np.percentile(diffs,97.5),(diffs<=0).mean()
print(f'\n[C] 점포 특성 모델 vs 기준선 (평가 2020–22 개업 {te.sum()}곳)')
print(f'  AUC: ① 프랜차이즈 여부 평균 {roc_auc_score(yt,pp):.3f} | ② 브랜드 평균 폐업률 {roc_auc_score(yt,pb):.3f} | 브랜드+면적 로지스틱 {roc_auc_score(yt,pl):.3f} | 후보 16개 GBM {roc_auc_score(yt,pg):.3f}')
for nm,a in [('브랜드+면적 로지스틱',pl),('후보 16개 GBM',pg)]:
    m,lo,hi,pv=boot(a,pb); print(f'  {nm} − ② 브랜드 평균: {m:+.3f} (95% {lo:+.3f}~{hi:+.3f}, 차이≤0 비율 {pv:.3f})')
# 개인 치킨집 안에서도 면적이 가르나 (브랜드 기준선은 개인 전부 같은 값)
ind=~d.loc[te,'프랜차이즈'].values
print(f'  개인 치킨집 안 면적 단독 AUC {1-roc_auc_score(yt[ind],d.loc[te,"면적"].values[ind]):.3f} (작을수록 폐업↑), 개인 {ind.sum()}곳')

# ---------------- D ----------------
print('\n[D] 네이버 리뷰·평점 파일로 폐업 예측이 되나')
rv=pd.read_csv(f'{O}/data/raw/막시무스/가공/치킨_네이버리뷰_동대문30분권.csv',low_memory=False)
st=pd.read_csv(f'{O}/data/raw/막시무스/가공/치킨점포_층_평점_v0.csv',low_memory=False)
print(f'  리뷰 {len(rv)}건, 점포 {rv.store_id.nunique()}곳, 업종: 치킨 {(rv.category_small=="치킨").mean():.1%}, 점포당 리뷰 최대 {rv.groupby("store_id").size().max()}건(상한 20), 별점 채움 {rv.rating.notna().mean():.1%}')
sn=rv.drop_duplicates('store_id').store_name.fillna('').str.replace(r'\s+','',regex=True)
ok=sn.str.contains(fi.cw.KW,regex=True)
for pat in fi.cw.B.values(): ok|=sn.str.contains(pat,regex=True)
print(f'  리뷰 점포 중 상호에 치킨 키워드·브랜드 없는 곳 {(~ok).sum()}/{len(sn)}곳, 예: {sn[~ok].head(10).tolist()}')
print(f'  점포·평점 파일 {len(st)}곳: 출처 {st.source.value_counts().to_dict()}, 네이버평점 있음 {st.네이버평점.notna().sum()}곳, 층 있음 {st.층.notna().sum()}곳')
# 인허가 치킨 표본(B_넓음 전 기간)과 30m 안 + 상호 앞 2글자 일치로 연결
c=fi.c; tf=Transformer.from_crs(4326,5174,always_xy=True)
T=cKDTree(c[['X','Y']].values); nm=c.nm.str.replace(r'[^가-힣A-Za-z0-9]','',regex=True)
def link(df,lat,lon,name):
    x,yv=tf.transform(df[lon].values,df[lat].values); out=[]
    for xi,yi,n in zip(x,yv,df[name].fillna('').str.replace(r'[^가-힣A-Za-z0-9]','',regex=True)):
        q=[j for j in T.query_ball_point([xi,yi],30) if n[:2] and nm.iat[j].startswith(n[:2])]
        out.append(q)
    return out
st['m']=link(st,'lat','lon','상호')
lk=st[st.m.str.len()>0]; closed=lk.m.apply(lambda q:c.close.iloc[q].notna().all())
op=lk.m.apply(lambda q:c.open.iloc[q].max())
print(f'  점포·평점 파일 → 인허가 연결 {len(lk)}/{len(st)}곳, 그중 인허가상 폐업 {closed.sum()}곳({closed.mean():.1%})')
print(f'  연결 점포 중 2015–22 개업(3년 폐업 y를 만들 수 있는 곳) {op.dt.year.between(2015,2022).sum()}곳, 그중 평점 있음 {lk[op.dt.year.between(2015,2022)].네이버평점.notna().sum()}곳')
e=lk[op.dt.year.between(2015,2022)].copy(); j=e.m.apply(lambda q:q[int(np.argmax(c.open.iloc[q].values))])
e['y3']=((c.close.iloc[j].values<=c.open.iloc[j].values+np.timedelta64(3*365,'D'))&~c.재등록.iloc[j].values).astype(int)
print(f'  그 542곳의 3년 실제 폐업 {e.y3.sum()}곳, 그중 평점 있음 {e[e.y3==1].네이버평점.notna().sum()}곳 | 인허가 폐업 114곳 중 평점 있음 {lk[closed].네이버평점.notna().sum()}곳')
print('  → 수집 시점(2026)에 살아 있는 점포 위주라 폐업 점포의 평점·리뷰가 거의 없음(생존 편향)')
