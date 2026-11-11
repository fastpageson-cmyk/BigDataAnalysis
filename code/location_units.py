# 입지 재검증 (치킨, chicken_wide.py B_넓음, y = 개업 3년 안 실제 폐업)
#  A. 지역 단위 비교: 행정동 / 공식 상권(밖은 동) / 격자 250·500·1000m / 음식점 밀집 군집(DBSCAN)
#     → 2010–17 개업의 단위별 폐업률(경험적 베이즈 수축)이 2018–22 개업의 폐업을 얼마나 가르는가
#  B. 반경 비교: 경쟁 C_r(영업 중 치킨집)·수요 D_r(저녁 생활인구, 행정동 밀도×버퍼 겹침 면적) r=250~2000m
#  C. 경로: 저녁 수요 → 치킨집 밀집(개업 시점) → 개업 후 2년 신규 진입 → 폐업
#  D. 이동 변수 재설계: 수요/경쟁 비, 거주자 비중(새벽/저녁), 30·40대 비중, 1년 증가율, 역 저녁 하차
import os,io,json,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
from sklearn.cluster import DBSCAN
from shapely.geometry import shape,Point
from shapely import STRtree,points
from shapely.ops import transform as stf
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
def load(name):
    spec=importlib.util.spec_from_file_location(name,f'{O}/code/{name}.py'); m=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign')
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year
c['y']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int); c['면적']=np.log1p(pd.to_numeric(c['소재지면적'],errors='coerce').fillna(40))

# 행정동 폴리곤(2020-07 경계, EPSG:5179→5174 변환) 점-폴리곤 배정
tf=Transformer.from_crs(5179,5174,always_xy=True).transform   # admdongkor geojson은 UTM-K(EPSG:5179) 좌표
g=json.load(open(f'{O}/data/raw/행정동경계/HangJeongDong_ver20200701.geojson')); F=[f for f in g['features'] if f['properties']['sido']=='11']
DG=[stf(tf,shape(f['geometry'])) for f in F]; DC=[f['properties']['adm_cd2'][:8] for f in F]; DT=STRtree(DG)
def pip(X,Y):
    idx=DT.query(points(X,Y),predicate='within'); out=np.full(len(X),None,object); out[idx[0]]=np.array(DC)[idx[1]]; return out
c['동']=pip(c.X.values,c.Y.values); print('행정동 배정률',c.동.notna().mean().round(3))
gi,_,_=aa.assign(c.X.values,c.Y.values); c['상권']=np.where(gi>=0,'A'+pd.Series(gi).astype(str),'out_'+c.동.astype(str))
for s in (250,500,1000): c[f'격자{s}']=(c.X//s).astype(int).astype(str)+'_'+(c.Y//s).astype(int).astype(str)
# 음식점 밀집 군집: 2017년 말 영업 중 음식점·카페 좌표 DBSCAN(eps 60m, 15곳), 군집 밖 치킨집은 동으로
a=cw.a; aX=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); aY=pd.to_numeric(a['좌표정보(Y)'],errors='coerce')
act=(a.open<=pd.Timestamp('2017-12-31'))&(a.close.isna()|(a.close>pd.Timestamp('2017-12-31')))&aX.notna()
pts=np.c_[aX[act],aY[act]]; lab=DBSCAN(eps=60,min_samples=15,algorithm='ball_tree').fit_predict(pts)
T=cKDTree(pts[lab>=0]); dist,ix=T.query(c[['X','Y']].values); L=lab[lab>=0][ix]
c['군집']=np.where(dist<=60,'K'+pd.Series(L).astype(str),'out_'+c.동.astype(str))
print(f'DBSCAN 군집 {lab.max()+1}개, 2017 음식점 {len(pts)}곳 중 군집 안 {(lab>=0).mean():.2f}, 치킨집 군집 안 {(dist<=60).mean():.2f}')

# ---------- A. 단위 비교 ----------
print('\n[A] 지역 단위별: 2010–17 개업 폐업률(수축, k=10) → 2018–22 개업 폐업 예측')
hist=c[c.연.between(2010,2017)]; test=c[c.연.between(2018,2022)].copy(); g0=hist.y.mean()
tr=c[c.연.between(2015,2017)].copy(); hist0=c[c.연.between(2010,2014)]
def rate(h,units,k=10):
    s=h.groupby(units).y.agg(['sum','size']); return ((s['sum']+k*h.y.mean())/(s['size']+k)), s['size']
rows=[]
for U in ['동','상권','격자250','격자500','격자1000','군집']:
    r,n=rate(hist,U); te=test.assign(rt=test[U].map(r).fillna(g0),nh=test[U].map(n).fillna(0))
    r0,_=rate(hist0,U); trn=tr.assign(rt=tr[U].map(r0).fillna(hist0.y.mean()))
    lg=lambda x:np.log(x/(1-x))
    m0=smf.logit('y~프랜차이즈+면적',trn).fit(disp=0); m1=smf.logit('y~프랜차이즈+면적+lg(rt)',trn).fit(disp=0)
    rows.append(dict(단위=U,단위수=c[U].nunique(),과거3곳이상=(te.nh>=3).mean(),AUC_단위율만=roc_auc_score(te.y,te.rt),
                     AUC_기본=roc_auc_score(te.y,m0.predict(te)),AUC_기본_단위=roc_auc_score(te.y,m1.predict(te))))
print(pd.DataFrame(rows).set_index('단위').round(3).to_string())

# ---------- B. 반경 비교 (2018–22 개업) ----------
lp=pd.read_csv(f'{P}/living_pop_dong_year.csv',dtype={'동':str}); area={k:gm.area for k,gm in zip(DC,DG)}
for v in ['새벽','저녁','저녁3049']: lp[v+'밀도']=lp[v]/lp.동.map(area)
LP={(r.동,r.연):r for r in lp.itertuples()}
d=test.copy(); ao=c.open.values; ac=c.close.fillna(pd.Timestamp('2100-01-01')).values; CT=cKDTree(c[['X','Y']].values)
R=[250,500,1000,1500,2000]
def buf_pop(x,y,yr,r,var):
    b=Point(x,y).buffer(r,16); tot=0.
    for i in DT.query(b):
        row=LP.get((DC[i],yr));
        if row is not None: tot+=getattr(row,var+'밀도')*DG[i].intersection(b).area
    return tot
for r in R:
    C_,E_,D_,Rs_,Y_,D2_=[],[],[],[],[],[]
    for x,y,o,yr in zip(d.X.values,d.Y.values,d.open.values,d.연.values):
        q=np.array(CT.query_ball_point([x,y],r),dtype=int)
        C_.append(((ao[q]<o)&(ac[q]>o)).sum()); E_.append(((ao[q]>o)&(ao[q]<=o+np.timedelta64(730,'D'))).sum())
        D_.append(buf_pop(x,y,yr-1,r,'저녁')); Rs_.append(buf_pop(x,y,yr-1,r,'새벽')); Y_.append(buf_pop(x,y,yr-1,r,'저녁3049'))
        D2_.append(buf_pop(x,y,yr-2,r,'저녁') if yr-2>=2017 else np.nan)
    d[f'C{r}']=C_; d[f'E{r}']=E_; d[f'D{r}']=D_; d[f'R{r}']=Rs_; d[f'Y{r}']=Y_; d[f'D2_{r}']=D2_
d=d[d[f'D{R[0]}']>0].copy()
trm=d.연<=2020; tem=~trm
print(f'\n[B] 반경별 (2018–22 개업 {len(d)}곳, 학습 2018–20 / 평가 2021–22). 값: 계수 OR(1 log 단위), p, 평가 AUC')
base='y~프랜차이즈+면적+C(연)'; rows=[]
for r in R:
    d[f'lC{r}']=np.log1p(d[f'C{r}']); d[f'lD{r}']=np.log(d[f'D{r}'])
    mC=smf.logit(f'{base}+lC{r}',d).fit(disp=0); mD=smf.logit(f'{base}+lD{r}',d).fit(disp=0); mB=smf.logit(f'{base}+lC{r}+lD{r}',d).fit(disp=0)
    mt=smf.logit(f'y~프랜차이즈+면적+lC{r}+lD{r}',d[trm]).fit(disp=0)
    rows.append(dict(반경=r,경쟁만_OR=np.exp(mC.params[f'lC{r}']),경쟁만_p=mC.pvalues[f'lC{r}'],수요만_OR=np.exp(mD.params[f'lD{r}']),수요만_p=mD.pvalues[f'lD{r}'],
                     둘다_경쟁OR=np.exp(mB.params[f'lC{r}']),둘다_수요OR=np.exp(mB.params[f'lD{r}']),둘다_수요p=mB.pvalues[f'lD{r}'],LL=mB.llf,평가AUC=roc_auc_score(d[tem].y,mt.predict(d[tem]))))
RB=pd.DataFrame(rows).set_index('반경'); print(RB.round(3).to_string())
m0=smf.logit('y~프랜차이즈+면적',d[trm]).fit(disp=0); print(f'  (입지 변수 없는 기본 평가 AUC {roc_auc_score(d[tem].y,m0.predict(d[tem])):.3f})')
rs=int(RB.LL.idxmax()); print(f'  → 우도 최대 반경 r*={rs}m')

# ---------- C. 경로: 수요 → 밀집 → 신규 진입 → 폐업 ----------
r=rs; d['lC']=d[f'lC{r}']; d['lD']=d[f'lD{r}']; d['lE']=np.log1p(d[f'E{r}'])
print(f'\n[C] 경로 (r={r}m)')
a1=smf.ols('lC~lD+C(연)',d).fit(); print(f'  수요→개업 시점 경쟁: 계수 {a1.params["lD"]:.2f} (p={a1.pvalues["lD"]:.3g}), 상관 {np.corrcoef(d.lD,d.lC)[0,1]:.2f}')
a2=smf.ols('lE~lD+lC+C(연)',d).fit(); print(f'  수요→개업 후 2년 신규 진입(경쟁 통제): 계수 {a2.params["lD"]:.2f} (p={a2.pvalues["lD"]:.3g})')
for nm,f in [('수요만(총효과)',f'{base}+lD'),('+개업 시점 경쟁',f'{base}+lD+lC'),('+개업 후 신규 진입',f'{base}+lD+lC+lE')]:
    m=smf.logit(f,d).fit(disp=0); s=' | '.join(f'{v} OR {np.exp(m.params[v]):.2f} p={m.pvalues[v]:.3f}' for v in ['lD','lC','lE'] if v in m.params)
    print(f'  {nm}: {s}')
d['수요5']=pd.qcut(d.lD,5,labels=['수요1(낮)','2','3','4','수요5(높)']); d['경쟁비']=d[f'C{r}']/(d[f'D{r}']/1e4)   # 저녁인구 1만 명당 치킨집
d['경쟁3']=d.groupby('수요5',observed=True).경쟁비.transform(lambda s:pd.qcut(s.rank(method='first'),3,labels=['경쟁 낮음','중간','경쟁 높음']))
t=d.groupby(['수요5','경쟁3'],observed=True).y.agg(['mean','size']); print('  수요 5분위 × (같은 수요 안에서) 1만명당 치킨집 3분위, 3년 실제 폐업률'); print(t['mean'].unstack().round(3).to_string()); print('  n:',t['size'].unstack().iloc[:,0].to_dict())

# ---------- D. 이동 변수 재설계 ----------
sm=pd.read_csv(f'{P}/subway_station_year.csv'); ev=[]
for x,y,yr in zip(d.X.values,d.Y.values,d.연.values):
    q=sm[sm.연==yr-1]; ev.append(q.저녁하차.values[np.hypot(q.X.values-x,q.Y.values-y)<=r].sum())
d['l역']=np.log1p(ev); d['거주비']=d[f'R{r}']/d[f'D{r}']; d['비중3049']=d[f'Y{r}']/d[f'D{r}']; d['l수요경쟁비']=np.log(d[f'D{r}']/(d[f'C{r}']+1))
d['증가']=np.log(d[f'D{r}']/d[f'D2_{r}'])
MOB='+l수요경쟁비+거주비+비중3049+l역'
print(f'\n[D] 재설계 변수 (r={r}m)')
m0=smf.logit(f'{base}+lC',d).fit(disp=0); m1=smf.logit(f'{base}+lC{MOB}',d).fit(disp=0); lr=2*(m1.llf-m0.llf)
print(f'  경쟁 통제 후 이동 4개 묶음 χ²={lr:.2f}, p={chi2.sf(lr,4):.3f}')
print('  '+pd.DataFrame({'OR':np.exp(m1.params),'p':m1.pvalues}).loc[['lC','l수요경쟁비','거주비','비중3049','l역']].round(3).to_string().replace('\n','\n  '))
dg=d[d.연>=2019]; g0_=smf.logit(f'{base}+lC',dg).fit(disp=0); g1=smf.logit(f'{base}+lC+증가',dg).fit(disp=0)
print(f'  저녁 수요 1년 증가율(2019–22 개업 {len(dg)}곳): OR {np.exp(g1.params["증가"]):.2f}, p={g1.pvalues["증가"]:.3f}')
for nm,f in [('기본',  'y~프랜차이즈+면적'),('+경쟁','y~프랜차이즈+면적+lC'),('+경쟁+이동4',f'y~프랜차이즈+면적+lC{MOB}')]:
    m=smf.logit(f,d[trm]).fit(disp=0); print(f'  평가 AUC {nm}: {roc_auc_score(d[tem].y,m.predict(d[tem])):.3f}')
f=d[d.프랜차이즈]; f=f[f.브랜드.isin(f.브랜드.value_counts()[lambda s:s>=20].index)]
b0=smf.logit('y~C(브랜드)+면적+C(연)+lC',f).fit(disp=0); b1=smf.logit(f'y~C(브랜드)+면적+C(연)+lC{MOB}',f).fit(disp=0); lr=2*(b1.llf-b0.llf)
print(f'  프랜차이즈 {len(f)}곳 브랜드 고정효과: 경쟁 OR {np.exp(b1.params["lC"]):.2f} p={b1.pvalues["lC"]:.3f}, 이동 묶음 χ²={lr:.2f} p={chi2.sf(lr,4):.3f}')
d.drop(columns=[x for x in d.columns if x in('지번주소','사업장명','nm','도로명주소')]).to_pickle(f'{P}/location_units_sample.pkl')
