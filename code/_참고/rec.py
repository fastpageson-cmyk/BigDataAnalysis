import sys, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
d=sys.argv[1]; R='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/'; RAW=R+'maximus-analysis-archive/data/raw/'
pd.set_option('display.width',220,'display.max_columns',40)
# crosswalk from public store snapshot: lot -> 행정동
sb=pd.read_csv(R+'데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv',usecols=['시군구명','행정동코드','행정동명','법정동명','지번본번지','지번부번지','경도','위도'],low_memory=False)
sb['gd']=sb['시군구명']+' '+sb['법정동명']; sb['bo']=sb['지번본번지'].astype('Int64').astype(str); sb['bu']=sb['지번부번지'].fillna(0).astype('Int64').astype(str)
def maj(keys):
    return sb.groupby(keys+['행정동코드']).size().reset_index(name='c').sort_values('c',ascending=False).drop_duplicates(keys)[keys+['행정동코드']]
c3,c2,c1=maj(['gd','bo','bu']),maj(['gd','bo']),maj(['gd'])
cols=['관리번호','인허가일자','영업상태명','폐업일자','소재지면적','지번주소','업태구분명']
df=pd.concat([pd.read_csv(f'{d}/{n}.csv',encoding='cp949',usecols=cols,dtype=str,encoding_errors='replace') for n in ['cafe','rest']]).drop_duplicates().drop_duplicates('관리번호',keep='last')
for c in ['인허가일자','폐업일자']: df[c]=pd.to_datetime(df[c].str.strip(),errors='coerce')
a=df['지번주소'].fillna('').str.extract(r'서울특별시\s+(\S+구)\s+(\S+?(?:동\d*가|동|가|로\d*가|로))\s+(?:산\s*)?(\d+)(?:-(\d+))?')
df['gd']=a[0]+' '+a[1]; df['bo']=a[2]; df['bu']=a[3].fillna('0'); n=len(df)
df=df[df.gd.notna()&~(df['영업상태명'].str.contains('폐업',na=False)&df['폐업일자'].isna())]
df=df[~df['업태구분명'].isin({'편의점','백화점','푸드트럭','철도역구내','키즈카페','기타 휴게음식점'})]
m=df.merge(c3.rename(columns={'행정동코드':'a3'}),how='left',on=['gd','bo','bu']).merge(c2.rename(columns={'행정동코드':'a2'}),how='left',on=['gd','bo']).merge(c1.rename(columns={'행정동코드':'a1'}),how='left',on='gd')
print('match exact',m.a3.notna().mean().round(3),'bon',(m.a3.isna()&m.a2.notna()).mean().round(3),'dong',(m.a3.isna()&m.a2.isna()&m.a1.notna()).mean().round(3),'none',(m.a3.isna()&m.a2.isna()&m.a1.isna()).mean().round(3),'of',len(m),'from',n)
m['adm']=m.a3.fillna(m.a2).fillna(m.a1); m=m[m.adm.notna()]; m['adm']=m.adm.astype('int64')
m['cafe']=m['업태구분명'].isin(['커피숍','까페','떡카페','전통찻집']); m['area']=pd.to_numeric(m['소재지면적'],errors='coerce')
names=sb.drop_duplicates('행정동코드').set_index('행정동코드')['행정동명']; gu=sb.drop_duplicates('행정동코드').set_index('행정동코드')['시군구명']; lon=sb.groupby('행정동코드')[['경도','위도']].median()
op=m['인허가일자']; cl=m['폐업일자'].fillna(pd.Timestamp('2099-1-1'))
recs=[]
for Y in range(2005,2027):
    e=pd.Timestamp(f'{Y}-06-30'); s=e-pd.DateOffset(years=1)
    x=m[(op<=e)&(cl>e)]; g=x.groupby('adm')
    r=pd.DataFrame({'n':g.size(),'cafe':g['cafe'].sum(),'area':g['area'].median()})
    r['opens']=m[(op>s)&(op<=e)].groupby('adm').size(); r['closes']=m[(cl>s)&(cl<=e)].groupby('adm').size(); r['Y']=Y
    recs.append(r.fillna({'opens':0,'closes':0}).reset_index())
p=pd.concat(recs).set_index(['adm','Y']).sort_index()
L=lambda c,h:p[c].groupby('adm').shift(h)
p['g1']=p.n/L('n',1)-1; p['g3']=p.n/L('n',3)-1; p['cg3']=(p.cafe+1)/(L('cafe',3)+1)-1; p['cafe_sh']=p.cafe/p.n; p['open_r']=p.opens/p.n; p['close_r']=p.closes/p.n; p['logn']=np.log(p.n); p['logcafe']=np.log1p(p.cafe)
p['gu']=p.index.get_level_values('adm').map(gu); p['gu_g3']=p.groupby(['gu','Y'])['g3'].transform('median')
ks=lon.index.to_numpy(); P=lon.to_numpy()*[88000,111000]; D=np.sqrt(((P[:,None]-P[None])**2).sum(2)); nb={k:ks[(D[i]<1500)&(D[i]>0)] for i,k in enumerate(ks)}
w=p.n.unstack('Y'); ng={}
for y in range(2008,2027):
    ng[y]=pd.Series({k:(lambda ne:(w.loc[ne,y].sum()/w.loc[ne,y-3].sum()-1) if len(ne) and w.loc[ne,y-3].sum()>0 else np.nan)([q for q in v if q in w.index]) for k,v in nb.items()})
p['near_g3']=pd.DataFrame(ng).rename_axis(columns='Y').stack().reindex(p.index)
BASE=['logn','logcafe','cafe_sh','g1','g3','cg3','open_r','close_r','area','gu_g3','near_g3']


import glob
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
S_DIR=sys.argv[2]
def rcsv(f):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,low_memory=False)
        except UnicodeDecodeError: pass
fr=[rcsv(f) for f in glob.glob(S_DIR+'/x3/*.csv')+glob.glob(S_DIR+'/x4/*.csv')+[RAW+f'seoul_commercial_sales_dong_{y}.csv' for y in (2023,2024)]+[d+'/sales.csv']]
sa=pd.concat(fr); sa['q']=sa['기준_년분기_코드'].astype(str)
print(sa[sa['서비스_업종_코드_명'].astype(str).str.contains('커피')][['서비스_업종_코드','서비스_업종_코드_명']].drop_duplicates().to_string())
sa=sa[sa['서비스_업종_코드_명'].astype(str).str.contains('커피')&sa.q.str.endswith('2')].drop_duplicates(['q','행정동_코드']); sa['Y']=sa.q.str[:4].astype(int)
print('coffee sales 조원',sa.groupby('Y')['당월_매출_금액'].sum().div(1e11).round(2).to_dict())
C=sa.set_index(['행정동_코드','Y'])[['당월_매출_금액','당월_매출_건수']].rename(columns={'당월_매출_금액':'csales','당월_매출_건수':'ctx'}).rename_axis(['adm','Y'])
def q2(f):
    x=rcsv(RAW+f); x=x[x['기준_년분기_코드'].astype(str).str.endswith('2')].copy(); x['Y']=x['기준_년분기_코드'].astype(str).str[:4].astype(int); return x.set_index(['행정동_코드','Y'])
fl,wk,rp,fc=q2('seoul_commercial_floating_population_dong.csv'),q2('seoul_workplace_population_dong.csv'),q2('seoul_resident_population_dong.csv'),q2('seoul_attracting_facilities_dong.csv')
E_=pd.DataFrame(index=fl.index); E_['lflow']=np.log1p(fl['총_유동인구_수']); E_['flow2030']=(fl['연령대_20_유동인구_수']+fl['연령대_30_유동인구_수'])/fl['총_유동인구_수']
E_['lwork']=np.log1p(wk['총_직장_인구_수']); E_['lres']=np.log1p(rp['총_상주인구_수']); E_['lfac']=np.log1p(fc['집객시설_수']); E_['subway']=fc['지하철_역_수'].fillna(0); E_=E_.rename_axis(['adm','Y'])
q=p[['n','cafe','logn','cg3','g3','gu']].join(C,how='inner').join(E_,how='left').reset_index()
q=q[(q.cafe>=5)&q.lflow.notna()&q.lwork.notna()&(q.csales>0)].copy(); q['name']=q.adm.map(names)
q['perc']=q.csales/q.cafe; q['lc']=np.log(q.cafe); q['lcs']=np.log(q.csales)
print('eligible',q.groupby('Y').size().to_dict())
DEM=['lflow','flow2030','lwork','lres','lfac','subway','logn']
def pct(s): return s.rank(pct=True)
def crossfit(df,y,F):
    out=np.zeros(len(df)); X=StandardScaler().fit_transform(df[F].values)
    for a,b in KFold(5,shuffle=True,random_state=0).split(X): out[b]=Ridge(alpha=1.0).fit(X[a],df[y].values[a]).predict(X[b])
    return out
XY=lon.copy(); XY['x']=XY['경도']*88000; XY['y']=XY['위도']*111000
def score(t,w=(.4,.3,.3),rcut=.6,inK=10):
    t=t.copy(); t['exp_s']=crossfit(t,'lcs',DEM); t['R']=pct(t.exp_s)   # 구조로 기대되는 커피 수요
    t['exp_c']=crossfit(t,'lc',DEM); t['gap_c']=t.lc-t.exp_c            # 기대보다 카페가 적을수록 음수
    t['exp_p']=crossfit(t,'lcs',DEM+['lc']); t['gap_s']=t.lcs-t.exp_p   # 카페 수까지 감안한 매출 잔차
    t['u_under']=pct(-t.gap_c); t['u_nov']=pct(-t.lcs)
    tops=set(t.nlargest(inK,'csales').adm)|set(t.nlargest(inK,'lflow').adm)|set(t.nlargest(inK,'cafe').adm)
    P=XY.loc[t.adm,['x','y']].values; Q=XY.loc[list(tops),['x','y']].values
    t['dist']=np.sqrt(((P[:,None]-Q[None])**2).sum(2)).min(1); t['u_sp']=pct(t.dist)
    t['U']=w[0]*t.u_under+w[1]*t.u_nov+w[2]*t.u_sp; t['S']=t.R*pct(t.U); t.loc[t.R<rcut,'S']=np.nan
    t['inertia']=t.adm.isin(tops); return t
def survive(T,h=2):
    e=pd.Timestamp(f'{T}-06-30'); s=e-pd.DateOffset(years=1); f=e+pd.DateOffset(years=h)
    x=m[m.cafe&(m['인허가일자']>s)&(m['인허가일자']<=e)]; cc=x['폐업일자'].fillna(pd.Timestamp('2099-1-1'))
    return (cc>f).groupby(x.adm).agg(['mean','size'])
allres=[]
def evalT(T,h=2,**k):
    t=score(q[q.Y==T],**k); v=q[q.Y==T+h].set_index('adm'); t=t.set_index('adm')
    t['p2']=v.perc.reindex(t.index); t['s2']=v.csales.reindex(t.index); t['perg']=t.p2/t.perc-1; t['salg']=t.s2/t.csales-1; t['c2']=v.cafe.reindex(t.index)
    sv=survive(T,h); t['surv']=sv['mean'].reindex(t.index); t['nnew']=sv['size'].reindex(t.index); t.loc[t.nnew<5,'surv']=np.nan
    lists={'의외성 S':t.nlargest(10,'S'),'R만':t.nlargest(10,'R'),'매출 상위':t.nlargest(10,'csales'),'유동 상위':t.nlargest(10,'lflow'),'카페수 상위':t.nlargest(10,'cafe'),'카페 증가율':t.nlargest(10,'cg3'),'카페당 매출':t.nlargest(10,'perc')}
    rng=np.random.default_rng(0); r=[]
    for nm,l in lists.items():
        pv=lambda c:(np.mean([np.nanmedian(t[c].values[rng.choice(len(t),10,replace=False)])>=np.nanmedian(l[c]) for _ in range(2000)]))
        r.append(dict(T=T,list=nm,perg=np.nanmedian(l.perg),salg=np.nanmedian(l.salg),surv=np.nanmedian(l.surv),nsv=l.surv.notna().sum(),perc_p=pv('perg'),surv_p=pv('surv')))
    r.append(dict(T=T,list='전체',perg=t.perg.median(),salg=t.salg.median(),surv=t.surv.median(),nsv=t.surv.notna().sum()))
    return pd.DataFrame(r),t
R=[];ts={}
for T in (2022,2023,2024):
    r,t=evalT(T); R.append(r); ts[T]=t
R=pd.concat(R); print(R.round(3).to_string(index=False))
print(R.groupby('list')[['perg','salg','surv']].mean().round(3).sort_values('surv',ascending=False).to_string())
for T in (2022,2023,2024):
    t=ts[T]; print(T,'corr gap_c vs surv',t[['gap_c','surv']].corr(method='spearman').iloc[0,1].round(3),'gap_c vs perg',t[['gap_c','perg']].corr(method='spearman').iloc[0,1].round(3),'gap_s vs perg',t[['gap_s','perg']].corr(method='spearman').iloc[0,1].round(3),'gap_s vs surv',t[['gap_s','surv']].corr(method='spearman').iloc[0,1].round(3),'R vs surv',t[['R','surv']].corr(method='spearman').iloc[0,1].round(3))
print('민감도')
for kw,lb in [(dict(w=(1/3,1/3,1/3)),'균등'),(dict(rcut=.5),'R0.5'),(dict(rcut=.7),'R0.7'),(dict(w=(1,0,0)),'공급부족만'),(dict(w=(.4,.6,0)),'공간제외')]:
    x=[evalT(T,**kw)[0] for T in (2022,2023,2024)]; x=pd.concat(x); x=x[x.list=='의외성 S']; print(lb,x[['perg','salg','surv']].mean().round(3).to_dict(), x.surv.round(3).tolist())
T=2026; t=score(q[q.Y==T]).set_index('adm'); t['gu']=t.index.map(gu)
print(t.nlargest(12,'S')[['gu','name','cafe','csales','perc','R','u_under','u_nov','u_sp','S','gap_c','inertia']].assign(csales=lambda d:(d.csales/1e8).round(1),perc=lambda d:(d.perc/1e6).round(1)).round(2).to_string(index=False))
print('관성 포함', t.inertia.sum(),'len',len(t),'S 있는 동',t.S.notna().sum())
for c in ['성수1가1동','성수1가2동','성수2가1동','성수2가3동']:
    print(t[t.name==c][['name','cafe','R','gap_c','S']].round(2).to_string(index=False,header=False))
