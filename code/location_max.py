# 입지로 폐업을 얼마나 예측할 수 있나: 가진 입지 정보를 전부 합쳐 한계를 잰다 (치킨, chicken_wide.py B_넓음)
#  [A] 개업 시점 예측: 2010–18 개업으로 학습 → 2019–22 개업의 3년 실제 폐업 예측. 변수 묶음을 하나씩 더하며 AUC 비교
#      경쟁(반경 100·250·500·1000m 치킨집, 같은 브랜드 거리), 주변 음식점(수·최근 2년 폐업률·폐업점 평균 수명),
#      자리 이력(같은 자리 30m의 이전 음식점 수·직전 점포 수명·공실 기간·직전이 치킨집인지), 지하철 출구 거리, 상권 유형
#  [B] 연 단위 위험 모형: 매년 1월 1일 영업 중인 점포가 그해 실제 폐업하나. 개업 후 바뀌는 주변(경쟁 증감·같은 브랜드 신규·주변 폐업)을 넣음
#      2015–20년으로 학습 → 2021–24년 예측
#  모든 변수는 판단 시점 이전 정보만 씀(누수 없음). 모형: HistGradientBoosting, 브랜드는 학습 구간 폐업률로 수축 인코딩
import os,io,json,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'; FAR=np.datetime64('2100-01-01')
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
bf=load('brand_fit'); cw=bf.cw; c=bf.c.reset_index(drop=True).copy()   # 공식유형·사람유형·면적 포함
c['y3']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
a=cw.a; al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close,'치킨':a.index.isin([])}).dropna(subset=['X','o']).reset_index(drop=True)
ao=al.o.values; ac=al.c.fillna(pd.Timestamp('2100-01-01')).values; AT=cKDTree(al[['X','Y']].values)
co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values; CT=cKDTree(c[['X','Y']].values); XY=c[['X','Y']].values
tf=Transformer.from_crs(4326,5174,always_xy=True)
e=json.load(open(f'{O}/data/raw/출입구/osm_subway_entrance_seoul.json'))['elements']; ex,ey=tf.transform([x['lon'] for x in e],[x['lat'] for x in e])
c['출구m']=cKDTree(np.c_[ex,ey]).query(XY)[0]
Y2=np.timedelta64(730,'D'); D1=np.timedelta64(365,'D')

def feats(idx,t):
    """점포 idx를 시점 t(각각) 기준으로 본 입지 변수"""
    f=pd.DataFrame(index=idx); xy=XY[idx]
    for r in (100,250,500,1000):
        f[f'치킨{r}']=[((co[q]<s)&(cc[q]>s)).sum()-((co[i]<s)&(cc[i]>s)) for i,q,s in zip(idx,CT.query_ball_point(xy,r),t)]
    br=c.브랜드.values; sb=[]; cl1=[]; op1=[]
    for i,q,s in zip(idx,CT.query_ball_point(xy,1000),t):
        q=np.array([j for j in q if j!=i],int); live=(co[q]<s)&(cc[q]>s)
        m=q[live&(br[q]==br[i])] if isinstance(br[i],str) else []
        sb.append(np.sqrt(((XY[m]-XY[i])**2).sum(1)).min() if len(m) else 1500)
    f['같은브랜드m']=sb
    q5=CT.query_ball_point(xy,500)
    f['치킨500_1y폐업']=[((cc[q]<s)&(cc[q]>=s-D1)).sum() for q,s in zip(q5,t)]; f['치킨500_1y개업']=[((co[q]<s)&(co[q]>=s-D1)).sum()-((co[i]<s)&(co[i]>=s-D1)) for i,q,s in zip(idx,q5,t)]
    n3=[];cr=[];lf=[]; h_n=[];h_last=[];h_gap=[];h_mean=[]
    for q,q30,s in zip(AT.query_ball_point(xy,300),AT.query_ball_point(xy,30),t):
        q=np.array(q,int); live=(ao[q]<s)&(ac[q]>s); dead=(ac[q]<s)&(ac[q]>=s-Y2)
        n3.append(live.sum()); cr.append(dead.sum()/max(live.sum()+dead.sum(),1)); lf.append(((ac[q][dead]-ao[q][dead])/np.timedelta64(365,'D')).mean() if dead.any() else np.nan)
        p=np.array(q30,int); p=p[ac[p]<s]   # 이 자리에서 t 전에 문 닫은 음식점
        h_n.append(len(p))
        if len(p): ten=(ac[p]-ao[p])/np.timedelta64(365,'D'); k=ac[p].argmax(); h_last.append(ten[k]); h_gap.append((s-ac[p][k])/np.timedelta64(365,'D')); h_mean.append(ten.mean())
        else: h_last.append(np.nan); h_gap.append(np.nan); h_mean.append(np.nan)
    f['음식점300']=n3; f['음식점300_2y폐업률']=cr; f['음식점300_폐업점수명']=lf
    f['자리_이전점포수']=h_n; f['자리_직전수명']=h_last; f['자리_공실년']=h_gap; f['자리_평균수명']=h_mean
    f['자리_직전치킨']=[int(((cc[q]<s)&(cc[q]>=s-Y2)).sum()-0>0) for q,s in zip(CT.query_ball_point(xy,30),t)]
    return f
def benc(tr,te,y,k=20):
    g=tr.groupby('브랜드')[y].agg(['sum','size']); p0=tr.loc[tr.프랜차이즈,y].mean(); m=((g['sum']+k*p0)/(g['size']+k))
    return [np.where(d.프랜차이즈,d.브랜드.map(m).fillna(p0),tr.loc[~tr.프랜차이즈,y].mean()) for d in (tr,te)]
def auc(tr,te,cols,y):
    m=HGB(max_depth=3,learning_rate=0.05,max_iter=200,min_samples_leaf=40,random_state=0).fit(tr[cols],tr[y]); return m.predict_proba(te[cols])[:,1]
def boot(y,p,n=500,seed=0):
    r=np.random.default_rng(seed); v=[]
    for _ in range(n):
        i=r.integers(0,len(y),len(y))
        if y[i].min()!=y[i].max(): v.append(roc_auc_score(y[i],p[i]))
    return np.percentile(v,[2.5,97.5])

# ---------- A. 개업 시점 ----------
d=c[c.연.between(2010,2022)].copy(); F=feats(d.index.values,d.open.values); d=d.join(F)
for v in ['공식유형','사람유형']: d[v+'_c']=d[v].astype('category').cat.codes
d['프랜']=d.프랜차이즈.astype(int); tr=d[d.연<=2018].copy(); te=d[d.연>=2019].copy(); tr['브랜드enc'],te['브랜드enc']=benc(tr,te,'y3')
G={'브랜드':['프랜','브랜드enc'],'점포':['면적'],'경쟁':['치킨100','치킨250','치킨500','치킨1000','같은브랜드m','치킨500_1y폐업','치킨500_1y개업'],
   '주변 음식점':['음식점300','음식점300_2y폐업률','음식점300_폐업점수명'],'자리 이력':['자리_이전점포수','자리_직전수명','자리_공실년','자리_평균수명','자리_직전치킨'],
   '역·상권유형':['출구m','공식유형_c','사람유형_c']}
LOC=G['경쟁']+G['주변 음식점']+G['자리 이력']+G['역·상권유형']
print(f'[A] 개업 시점 예측: 학습 2010–18 개업 {len(tr)}곳 → 평가 2019–22 개업 {len(te)}곳 (3년 실제 폐업률 {te.y3.mean():.3f}, 프랜차이즈 {te.프랜.sum()}곳)')
sets=[('브랜드만',G['브랜드']),('브랜드+점포 면적',G['브랜드']+G['점포'])]+[(f'  + {k}',G['브랜드']+G['점포']+G[k]) for k in ['경쟁','주변 음식점','자리 이력','역·상권유형']]+[('브랜드+점포+입지 전부',G['브랜드']+G['점포']+LOC),('입지만(브랜드 없이)',LOC)]
yt=te.y3.values; fr=te.프랜.values==1; rows=[]
for nm,cols in sets:
    p=auc(tr,te,cols,'y3'); lo,hi=boot(yt,p)
    rows.append(dict(변수=nm,AUC=roc_auc_score(yt,p),하한=lo,상한=hi,프랜차이즈안=roc_auc_score(yt[fr],p[fr]),개인안=roc_auc_score(yt[~fr],p[~fr])))
print(pd.DataFrame(rows).round(3).to_string(index=False))
p0=auc(tr,te,G['브랜드']+G['점포'],'y3'); p1=auc(tr,te,G['브랜드']+G['점포']+LOC,'y3'); r=np.random.default_rng(1); dv=[]
for _ in range(1000):
    i=r.integers(0,len(yt),len(yt)); dv.append(roc_auc_score(yt[i],p1[i])-roc_auc_score(yt[i],p0[i]))
print(f'  입지를 더했을 때 AUC 변화 {np.mean(dv):+.3f} [95% {np.percentile(dv,2.5):+.3f}, {np.percentile(dv,97.5):+.3f}]')
te['p']=p1; te['구간']=pd.qcut(te.p,5,labels=['가장 안전','','중간',' ','가장 위험']); print('  전체 모형 예측 5분위별 실제 폐업률:',te.groupby('구간').y3.mean().round(3).tolist())
print('  자리 이력 단독 비교(평가 구간):'); te['이전있음']=te.자리_이전점포수>0
print('   이전 음식점 없던 자리 vs 있던 자리:',te.groupby('이전있음').y3.agg(['mean','size']).round(3).to_dict('index'))
h=te[te.이전있음].copy(); h['직전수명구간']=pd.cut(h.자리_직전수명,[0,1,3,6,100],labels=['1년 미만','1–3년','3–6년','6년 이상']); print('   직전 점포 수명별:',h.groupby('직전수명구간').y3.agg(['mean','size']).round(3).to_dict('index'))

import statsmodels.formula.api as smf
z=d[d.자리_이전점포수>0].copy(); z['l직전수명']=np.log(z.자리_직전수명.clip(lower=0.1)); z['직전1년미만']=(z.자리_직전수명<1).astype(int)
for nm,g in [('전체',z),('프랜차이즈',z[z.프랜==1]),('개인',z[z.프랜==0])]:
    m=smf.logit('y3~프랜+면적+C(연)+np.log1p(치킨500)+직전1년미만' if nm=='전체' else 'y3~면적+C(연)+np.log1p(치킨500)+직전1년미만',g).fit(disp=0)
    print(f'   직전 점포가 1년 못 버틴 자리(2010–22 개업, 연도·면적·경쟁 통제) {nm} n={len(g)}: OR {np.exp(m.params["직전1년미만"]):.2f}, p={m.pvalues["직전1년미만"]:.4f}, 폐업률 {g[g.직전1년미만==1].y3.mean():.3f} vs {g[g.직전1년미만==0].y3.mean():.3f}')

# ---------- B. 연 단위 위험 모형 ----------
rows=[]
for yr in range(2015,2025):
    s=np.datetime64(f'{yr}-01-01'); live=np.where((co<s)&(cc>s))[0]; f=feats(live,np.full(len(live),s))
    f['연도']=yr; f['나이']=(s-co[live])/np.timedelta64(365,'D'); f['y']=((cc[live]<np.datetime64(f'{yr+1}-01-01'))&~c.재등록.values[live]).astype(int)
    f['치킨500_3y변화']=f.치킨500-feats(live,np.full(len(live),s-np.timedelta64(1095,'D'))).치킨500
    f['같은브랜드1y신규']=[int(any(isinstance(c.브랜드.values[i],str) and c.브랜드.values[j]==c.브랜드.values[i] and co[j]>=s-D1 and co[j]<s for j in q if j!=i)) for i,q in zip(live,CT.query_ball_point(XY[live],1000))]
    rows.append(f.join(c.loc[live,['브랜드','프랜차이즈','면적','공식유형','사람유형','출구m']]))
H=pd.concat(rows); H['프랜']=H.프랜차이즈.astype(int)
for v in ['공식유형','사람유형']: H[v+'_c']=H[v].astype('category').cat.codes
tr=H[H.연도<=2020].copy(); te=H[H.연도>=2021].copy(); tr['브랜드enc'],te['브랜드enc']=benc(tr,te,'y')
B0=['프랜','브랜드enc','면적','나이']; STAT=['치킨100','치킨250','치킨500','치킨1000','같은브랜드m','음식점300','출구m','공식유형_c','사람유형_c','자리_이전점포수','자리_평균수명']
DYN=['치킨500_1y폐업','치킨500_1y개업','치킨500_3y변화','같은브랜드1y신규','음식점300_2y폐업률','음식점300_폐업점수명']
print(f'\n[B] 연 단위 위험 모형: 학습 2015–20 점포-연 {len(tr)}개 → 평가 2021–24 {len(te)}개 (연 실제 폐업률 {te.y.mean():.3f})')
yt=te.y.values; fr=te.프랜.values==1; P={}
for nm,cols in [('브랜드+면적+영업 연차',B0),('  + 지금 주변 상태',B0+STAT),('  + 최근 주변 변화',B0+STAT+DYN),('주변 상태·변화만',STAT+DYN)]:
    p=auc(tr,te,cols,'y'); P[nm]=p; lo,hi=boot(yt,p,200)
    print(f'  {nm}: AUC {roc_auc_score(yt,p):.3f} [{lo:.3f}–{hi:.3f}], 프랜차이즈 안 {roc_auc_score(yt[fr],p[fr]):.3f}, 개인 안 {roc_auc_score(yt[~fr],p[~fr]):.3f}')
te['p']=P['  + 최근 주변 변화']; te['구간']=pd.qcut(te.p,10,labels=False); print('  전체 모형 예측 10분위별 연 폐업률:',te.groupby('구간').y.mean().round(3).tolist())
print('  최근 변화 변수별 연 폐업률(평가 구간):')
for v,cut in [('치킨500_1y폐업',[-1,0,2,100]),('치킨500_1y개업',[-1,0,2,100]),('같은브랜드1y신규',[-1,0,1])]:
    print(f'   {v}:',te.groupby(pd.cut(te[v],cut)).y.agg(['mean','size']).round(4).to_dict('index'))
