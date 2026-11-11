# 입지 재검증 2: 긴 기간·사람/이동 특성 군집·KNN 유사 상권 (치킨, chicken_wide.py B_넓음)
#  1. 기간: 폐업 기준 3년 vs 5년 × 과거 이력 5·10·15년. 이력은 결과가 평가 시작(2017-01) 전에 다 관측된 개업만 씀(누수 없음)
#  2. 상권 유형 군집: 공식 상권 1,650개를 유동인구(시간대·연령·주말)·상주·직장인구·음식점 구성으로 KMeans, 상권 밖은 '비상권' 유형
#     → 유형 고정효과, 브랜드×유형 궁합(프랜차이즈) 우도비검정
#  3. KNN 유사 상권: 평가 점포마다 과거 점포 중 (a) 거리상 가까운 k곳 (b) 특성이 비슷한 k곳의 폐업률 → 예측력 비교
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score,silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'; Rw=f'{O}/data/raw'
def load(name):
    spec=importlib.util.spec_from_file_location(name,f'{O}/code/{name}.py'); m=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign')
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year; c['면적']=np.log1p(pd.to_numeric(c['소재지면적'],errors='coerce').fillna(40))
for H in (3,5): c[f'y{H}']=((c.close<=c.open+pd.DateOffset(years=H))&~c.재등록).astype(int)
gi,_,_=aa.assign(c.X.values,c.Y.values); c['상권i']=gi
c['격자500']=(c.X//500).astype(int).astype(str)+'_'+(c.Y//500).astype(int).astype(str)
A=aa.A   # area_assign이 shp에서 바로 만든 상권 표(assign()의 인덱스와 같은 순서)
c['상권']=np.where(gi>=0,'A'+pd.Series(gi).astype(str),'out_'+c.격자500)
lg=lambda x:np.log(x/(1-x))
TE=c[c.연.between(2017,2021)].copy()   # 평가: 2017–21 개업(5년 결과가 2026-11 전에 관측됨)
print(f'평가 점포 2017–21 개업 {len(TE)}곳, 5년 실제 폐업률 {TE.y5.mean():.3f}, 3년 {TE.y3.mean():.3f}')

# ---------- 1. 기간 ----------
print('\n[1] 과거 이력 단위 폐업률 → 2017–21 개업 폐업 (AUC). 이력 개업 구간은 결과가 2016년까지 관측되는 범위')
rows=[]
for H in (3,5):
    for W in (5,10,15):
        lo,hi=2016-H-W+1,2016-H; h=c[c.연.between(lo,hi)]; g0=h[f'y{H}'].mean()
        for U in ('상권','격자500'):
            s=h.groupby(U)[f'y{H}'].agg(['sum','size']); rt=(s['sum']+10*g0)/(s['size']+10)
            te=TE.assign(rt=TE[U].map(rt).fillna(g0)); m=smf.logit(f'y{H}~프랜차이즈+면적+C(연)+lg(rt)',te).fit(disp=0)
            rows.append(dict(폐업기준=f'{H}년',이력=f'{W}년({lo}–{hi})',단위=U,이력점포=len(h),AUC_단위율=roc_auc_score(te[f'y{H}'],te.rt),
                             단위율_OR=np.exp(m.params['lg(rt)']),p=m.pvalues['lg(rt)']))
print(pd.DataFrame(rows).round(3).to_string(index=False))

# ---------- 2. 상권 유형 군집 (사람·이동·상권 특성) ----------
def qavg(f,pre):  # 2021–22 분기 평균(평가 기간과 겹치지만 상권 구조 특성으로 봄)
    d=pd.read_csv(f'{Rw}/막시무스/상권/{f}',encoding='utf-8-sig'); d=d[d.STDR_YYQU_CD.between(20211,20224)]
    return d.groupby('TRDAR_CD').mean(numeric_only=True)
fl=qavg('유동인구_상권_2021_2026.csv','FL'); rs=qavg('상주인구_상권.csv','RE'); wk=qavg('직장인구_상권.csv','WK')
F=pd.DataFrame(index=fl.index); tot=fl.TOT_FLPOP_CO
for t in ['00_06','06_11','11_14','14_17','17_21','21_24']: F[f'유동_{t}']=fl[f'TMZON_{t}_FLPOP_CO']/tot
for a_ in ['20','30','40','50','60_ABOVE']: F[f'유동_{a_}대']=fl[f'AGRDE_{a_}_FLPOP_CO']/tot
F['주말비']=(fl.SAT_FLPOP_CO+fl.SUN_FLPOP_CO)/tot
A2=A.assign(TRDAR_CD=pd.to_numeric(A.상권코드)).set_index('TRDAR_CD'); ha=A2.면적m2/1e4
F['l유동밀도']=np.log1p(tot/ha); F['l상주밀도']=np.log1p(rs.TOT_REPOP_CO.reindex(F.index).fillna(0)/ha)
F['l직장밀도']=np.log1p(wk.TOT_WRC_POPLTN_CO.reindex(F.index).fillna(0)/ha)   # 아파트 가구 수(APT_HSHLD_CO)는 원천이 전부 0이라 제외
# 음식점 구성: 2019년 말 영업 중 인허가(카페·호프·기타) 상권별
a=cw.a; aX=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); aY=pd.to_numeric(a['좌표정보(Y)'],errors='coerce')
act=(a.open<=pd.Timestamp('2019-12-31'))&(a.close.isna()|(a.close>pd.Timestamp('2019-12-31')))&aX.notna()
ag,_,_=aa.assign(aX[act].values,aY[act].values); kind=np.where(a.src[act]=='cafe','카페',np.where(a.업태구분명[act]=='호프/통닭','호프','기타'))
mix=pd.crosstab(ag,kind); mix=mix[mix.index>=0]; mix.index=A.상권코드.astype(int).values[mix.index]
F['l음식점밀도']=np.log1p(mix.sum(axis=1).reindex(F.index).fillna(0)/ha); F['카페비']=(mix.카페/mix.sum(axis=1)).reindex(F.index).fillna(0); F['호프비']=(mix.호프/mix.sum(axis=1)).reindex(F.index).fillna(0)
F=F.dropna(); Z=StandardScaler().fit_transform(F)
sil={k:silhouette_score(Z,KMeans(k,n_init=20,random_state=0).fit_predict(Z)) for k in range(3,9)}; K=max(sil,key=sil.get)
km=KMeans(K,n_init=20,random_state=0).fit(Z); F['유형']=km.labels_
print(f'\n[2] 상권 유형 군집: 상권 {len(F)}개, 실루엣 {dict((k,round(v,3)) for k,v in sil.items())} → k={K}')
prof=F.groupby('유형').mean(); prof['상권수']=F.groupby('유형').size()
show=['상권수','유동_17_21','유동_21_24','유동_20대','유동_30대','주말비','l유동밀도','l상주밀도','l직장밀도','l음식점밀도','카페비','호프비']
print(prof[show].round(2).to_string())
code2type=F.유형.to_dict(); c['유형']=[('T'+str(code2type.get(int(A.상권코드.iat[i]),'?'))) if i>=0 else '비상권' for i in c.상권i]
# 유형별 5년 폐업(2010–21 개업) + 유형 고정효과·브랜드×유형 궁합
d=c[c.연.between(2010,2021)].copy()
print('\n유형별 2010–21 개업 5년 실제 폐업률 (프랜차이즈/개인)'); print(d.groupby(['유형','프랜차이즈']).y5.agg(['mean','size']).unstack().round(3).to_string())
m0=smf.logit('y5~프랜차이즈+면적+C(연)',d).fit(disp=0); m1=smf.logit('y5~프랜차이즈+면적+C(연)+C(유형)',d).fit(disp=0); lr=2*(m1.llf-m0.llf); df_=len(m1.params)-len(m0.params)
print(f'유형 고정효과 χ²={lr:.1f}, df={df_}, p={chi2.sf(lr,df_):.4f}')
tr=d.연<=2016; te=~tr
for nm,f in [('기본','y5~프랜차이즈+면적'),('+유형','y5~프랜차이즈+면적+C(유형)')]:
    m=smf.logit(f,d[tr]).fit(disp=0); print(f'  평가 AUC(2010–16 학습 → 2017–21) {nm}: {roc_auc_score(d[te].y5,m.predict(d[te])):.3f}')
f=d[d.프랜차이즈]; f=f[f.브랜드.isin(f.브랜드.value_counts()[lambda s:s>=40].index)]
b0=smf.logit('y5~C(브랜드)+C(유형)+면적+C(연)',f).fit(disp=0,method='bfgs',maxiter=500); b1=smf.logit('y5~C(브랜드)*C(유형)+면적+C(연)',f).fit(disp=0,method='bfgs',maxiter=2000)
lr=2*(b1.llf-b0.llf); df_=len(b1.params)-len(b0.params); print(f'브랜드×유형 궁합(프랜차이즈 {len(f)}곳, 브랜드 {f.브랜드.nunique()}개): χ²={lr:.1f}, df={df_}, p={chi2.sf(lr,df_):.3f}')
print('  브랜드×유형 5년 폐업률(점포 15곳 이상 칸)'); bt=f.groupby(['브랜드','유형']).y5.agg(['mean','size']); print(bt[bt['size']>=15]['mean'].unstack().round(2).to_string())

# ---------- 3. KNN 유사 상권 ----------
# 점포 특성(개업 시점): 500m 영업 중 치킨집·음식점 수, 카페·호프 비율 + 소속 상권 유형 특성(상권 밖은 0과 '비상권' 표시)
CT=cKDTree(c[['X','Y']].values); co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values
all_=pd.DataFrame({'X':aX,'Y':aY,'o':a.open,'c':a.close.fillna(pd.Timestamp('2100-01-01')),'k':np.where(a.src=='cafe',1,np.where(a.업태구분명=='호프/통닭',2,0))}).dropna(subset=['X'])
AT=cKDTree(all_[['X','Y']].values); ao=all_.o.values; ac=all_.c.values; ak=all_.k.values
H=5; hist=c[c.연.between(2016-H-10+1,2016-H)].copy(); pool=pd.concat([hist,TE])
feat=[]
for x,y,o in zip(pool.X.values,pool.Y.values,pool.open.values):
    q=np.array(CT.query_ball_point([x,y],500),dtype=int); nc=((co[q]<o)&(cc[q]>o)).sum()
    q=np.array(AT.query_ball_point([x,y],500),dtype=int); m=(ao[q]<o)&(ac[q]>o); n=m.sum(); feat.append((nc,n,(ak[q][m]==1).mean() if n else 0,(ak[q][m]==2).mean() if n else 0))
pool[['치킨500','음식점500','카페비500','호프비500']]=feat
FZ=pd.DataFrame(Z,index=F.index,columns=F.columns.drop('유형'))
tz=np.vstack([FZ.loc[int(A.상권코드.iat[i])].values if (i>=0 and int(A.상권코드.iat[i]) in FZ.index) else np.zeros(FZ.shape[1]) for i in pool.상권i])
X=np.c_[np.log1p(pool[['치킨500','음식점500']].values),pool[['카페비500','호프비500']].values,(pool.상권i<0).values]; X=StandardScaler().fit_transform(X)
X=np.c_[X,tz*0.6]   # 상권 사람·이동 특성(15개)이 점포 특성(5개)을 압도하지 않게 가중
ih=(pool.연<=2016-H).values; it=~ih; yh=pool.y5.values[ih]
print(f'\n[3] KNN 유사 상권: 과거 점포 {ih.sum()}곳(2002–11 개업, 5년 결과 2016년까지 관측) → 평가 {it.sum()}곳(2017–21)')
G=cKDTree(pool[['X','Y']].values[ih]); FT=cKDTree(X[ih]); te=pool[it].copy(); rows=[]
for k in (30,100,300):
    te[f'거리k{k}']=yh[G.query(pool[['X','Y']].values[it],k)[1]].mean(axis=1); te[f'특성k{k}']=yh[FT.query(X[it],k)[1]].mean(axis=1)
    for kind_ in ('거리','특성'):
        v=te[f'{kind_}k{k}'].clip(.01,.99); m=smf.logit('y5~프랜차이즈+면적+C(연)+lg(v)',te.assign(v=v)).fit(disp=0)
        rows.append(dict(이웃=kind_,k=k,AUC_이웃률=roc_auc_score(te.y5,v),OR=np.exp(m.params['lg(v)']),p=m.pvalues['lg(v)']))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print(f'  (참고) 기본(프랜차이즈·면적) 평가 AUC: {roc_auc_score(te.y5,smf.logit("y5~프랜차이즈+면적",pool[ih]).fit(disp=0).predict(te)):.3f}')
F.to_csv(f'{P}/area_types_people.csv',encoding='utf-8-sig')
