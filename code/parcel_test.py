# 필지 정보(공시지가·건축물대장·층)가 치킨집 3년 실제 폐업을 가르나 (parcel_fetch.py 결과 사용)
#  변수: 개업 전년 공시지가(원/㎡, 같은 해 서울 안 백분위), 건물 나이·지상층수·연면적·주용도·한 필지 건물 수, 주소에 적힌 층(1층/지하/2층 이상)
#  [1] 변수별 폐업률표 + 로지스틱(프랜차이즈·면적·연도·500m 경쟁 통제), 프랜차이즈/개인
#  [2] 시간 분할 예측: 학습 2010–18 → 평가 2019–22 개업. 브랜드+면적 vs +필지 정보 vs +필지+자리 이력 (AUC, 부트스트랩)
import os,io,re,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/필지'
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
pf=load('parcel_fetch'); sh=load('site_history')
c=pf.parcels().reset_index(drop=True); c['연']=c.open.dt.year; c['y3']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
c['면적']=np.log1p(pd.to_numeric(c.소재지면적,errors='coerce').fillna(40)); c['프랜']=c.프랜차이즈.astype(int)
K=['구','동','산','본','부','기준연']
J=pd.read_csv(f'{R}/jiga.csv',dtype=str); J=J[J.ok=='True'].drop_duplicates(K)
c[K]=c[K].astype(str); c=c.merge(J[K+['공시지가','시군구코드','법정동코드']],on=K,how='left')
c['공시지가']=pd.to_numeric(c.공시지가,errors='coerce'); c['l지가']=np.log(c.공시지가)
c['지가백분위']=c.groupby('기준연').l지가.rank(pct=True)
HASB=os.path.exists(f'{R}/brtitle.csv') and len(pd.read_csv(f'{R}/brtitle.csv',dtype=str))>0
if HASB:
  B=pd.read_csv(f'{R}/brtitle.csv',dtype=str); B=B[B.get('건물없음','nan').astype(str)!='1']; B=B[B.get('조회실패','nan').astype(str)!='1']
  for k in ['grndFlrCnt','ugrndFlrCnt','totArea']: B[k]=pd.to_numeric(B[k],errors='coerce')
  B['승인연']=pd.to_numeric(B.useAprDay.astype(str).str[:4],errors='coerce').where(lambda x:x>1900)
  B['용도']=np.select([B.mainPurpsCdNm.str.contains('근린생활',na=False),B.mainPurpsCdNm.str.contains('주택',na=False),B.mainPurpsCdNm.str.contains('업무|판매',na=False)],['근생','주거','업무판매'],'기타')
  B['key']=B.sigunguCd+'_'+B.bjdongCd+'_'+B.platGbCd.str[0].map({'0':'0','1':'1'}).fillna('0')+'_'+B.bun.astype(str).str.lstrip('0').replace('','0')+'_'+B.ji.astype(str).str.lstrip('0').replace('','0')
  main=B[B.mainAtchGbCdNm.fillna('주건축물')=='주건축물']
  g=main.sort_values('totArea',ascending=False).groupby('key')
  bf=pd.DataFrame({'건물수':g.size(),'지상층':g.grndFlrCnt.max(),'연면적':g.totArea.sum(),'승인연':g.승인연.min(),'용도':g.용도.first()})
if HASB:
  c['key']=c.시군구코드.astype(str)+'_'+c.법정동코드.astype(str)+'_'+c.산.astype(str)+'_'+c.본.astype(str).str.lstrip('0').replace('','0')+'_'+c.부.astype(str).str.lstrip('0').replace('','0')
  c=c.join(bf,on='key'); c['건물나이']=(c.연-c.승인연).where(lambda x:x>=0)
for k in ['건물수','지상층','연면적','승인연','용도','건물나이']:
    if k not in c: c[k]=np.nan
a=c.지번주소.fillna('')
fl=np.select([a.str.contains(r'지하|B\d|지\d층'),a.str.contains(r'(?<!\d)1\s*층|1층|\s1\d\d호'),a.str.contains(r'(?<!\d)[2-9]\s*층|[2-9]\d\d호|\d{2}층')],['지하','1층','2층 이상'],'표기 없음')
c['층']=fl
s=sh.site(c[['X','Y']].values,c.open.values,30); c['자리_직전수명']=s.직전수명.values; c['자리_단명5y']=s.단명5y.values; c['회전자리']=((s.직전수명<1)|(s.단명5y>=2)).astype(int).values
co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values; T=cKDTree(c[['X','Y']].values)
c['치킨500']=[((co[q]<o)&(cc[q]>o)).sum() for q,o in zip(T.query_ball_point(c[['X','Y']].values,500),co)]
d=c[c.연.between(2010,2022)].copy()
print(f'[0] 치킨집 2010–22 개업 {len(d)}곳: 공시지가 붙음 {d.공시지가.notna().mean():.3f}, 건축물대장 붙음 {d.지상층.notna().mean():.3f}, 주소에 층 표기 {(d.층!="표기 없음").mean():.3f}')

def tab(v,bins=None,labels=None):
    x=pd.cut(d[v],bins,labels=labels,right=False) if bins else d[v]
    t=d.groupby([x,'프랜']).y3.agg(['mean','size']).unstack(); return ', '.join(f'{k}: 개인 {r[("mean",0)]:.3f}({int(r[("size",0)])}) 프랜 {r[("mean",1)]:.3f}({int(r[("size",1)])})' for k,r in t.iterrows())
print('\n[1] 변수별 3년 실제 폐업률')
print('  공시지가 백분위:',tab('지가백분위',[0,.25,.5,.75,1.01],['하위25%','25–50%','50–75%','상위25%']))
if HASB: print('  건물 나이:',tab('건물나이',[0,10,20,30,200],['10년 미만','10–20년','20–30년','30년 이상']))
if HASB: print('  지상층수:',tab('지상층',[0,3,5,10,200],['1–2층','3–4층','5–9층','10층 이상']))
print('  주소상 층:',tab('층'))
if HASB: print('  주용도:',tab('용도'))
print('  통제 로지스틱 (면적·연도·log 경쟁, 전체엔 프랜차이즈도):')
for v,f in [('지가백분위','지가백분위'),('층','C(층,Treatment("1층"))')]+([('건물나이','np.log1p(건물나이)'),('지상층','np.log1p(지상층)'),('용도','C(용도,Treatment("근생"))')] if HASB else []):
    out=[]
    for nm,g in [('전체',d),('개인',d[d.프랜==0]),('프랜',d[d.프랜==1])]:
        m=smf.logit(f'y3~면적+C(연)+np.log1p(치킨500)+{f}'+('+프랜' if nm=='전체' else ''),g.dropna(subset=[v])).fit(disp=0)
        ks=[k for k in m.params.index if v in k]; out.append(f'{nm} '+' '.join(f'{k.split("[")[-1].rstrip("]").replace("T.","")} OR {np.exp(m.params[k]):.2f}(p={m.pvalues[k]:.3f})' for k in ks))
    print(f'   {v}: '+' | '.join(out))

tr=d[d.연<=2018].copy(); te=d[d.연>=2019].copy()
gb=tr[tr.프랜==1].groupby('브랜드').y3.agg(['sum','size']); p0=tr[tr.프랜==1].y3.mean(); enc=(gb['sum']+20*p0)/(gb['size']+20)
for x in (tr,te): x['브랜드enc']=np.where(x.프랜==1,x.브랜드.map(enc).fillna(p0),tr[tr.프랜==0].y3.mean())
for x in (tr,te): x['용도c']=x.용도.map({'근생':0,'주거':1,'업무판매':2,'기타':3}); x['층c']=x.층.map({'1층':0,'지하':1,'2층 이상':2,'표기 없음':3})
BASE=['프랜','브랜드enc','면적']; PAR=['지가백분위','층c']+(['건물나이','지상층','연면적','건물수','용도c'] if HASB else []); SITE=['자리_직전수명','자리_단명5y','치킨500']
yt=te.y3.values; fr=te.프랜.values==1; P={}
print(f'\n[2] 시간 분할 예측: 학습 2010–18 {len(tr)}곳 → 평가 2019–22 {len(te)}곳')
for nm,cols in [('브랜드+면적',BASE),('+ 필지 정보',BASE+PAR),('+ 필지 + 자리 이력·경쟁',BASE+PAR+SITE),('필지 정보만',PAR)]:
    p=HGB(max_depth=3,learning_rate=0.05,max_iter=200,min_samples_leaf=40,random_state=0).fit(tr[cols],tr.y3).predict_proba(te[cols])[:,1]; P[nm]=p
    print(f'  {nm}: AUC {roc_auc_score(yt,p):.3f}, 프랜차이즈 안 {roc_auc_score(yt[fr],p[fr]):.3f}, 개인 안 {roc_auc_score(yt[~fr],p[~fr]):.3f}')
r=np.random.default_rng(0); dv=[]
for _ in range(1000):
    i=r.integers(0,len(yt),len(yt)); dv.append(roc_auc_score(yt[i],P['+ 필지 + 자리 이력·경쟁'][i])-roc_auc_score(yt[i],P['브랜드+면적'][i]))
print(f'  필지+자리 이력 추가 AUC 변화 {np.mean(dv):+.3f} [95% {np.percentile(dv,2.5):+.3f}, {np.percentile(dv,97.5):+.3f}]')
te['p']=P['+ 필지 + 자리 이력·경쟁']; print('  예측 5분위별 실제 폐업률:',te.groupby(pd.qcut(te.p,5,labels=False)).y3.mean().round(3).tolist())
d[['연','프랜','y3','지가백분위','건물나이','지상층','용도','층','회전자리']].assign(구=d.구).groupby(['구']).agg(점포=('y3','size'),폐업률=('y3','mean'),지가백분위=('지가백분위','median'),건물나이=('건물나이','median')).round(3).to_csv(f'{O}/data/processed/parcel_by_gu.csv',encoding='utf-8-sig')
