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
S_DIR=sys.argv[2]; RAWD=RAW
def rcsv(f):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,low_memory=False)
        except UnicodeDecodeError: pass
fr=[rcsv(f) for f in glob.glob(S_DIR+'/x3/*.csv')+glob.glob(S_DIR+'/x4/*.csv')+[RAWD+f'seoul_commercial_sales_dong_{y}.csv' for y in (2023,2024)]+[d+'/sales.csv']]
sa=pd.concat(fr); sa['q']=sa['기준_년분기_코드'].astype(str); print(sorted(sa.q.unique()))
sa=sa[sa['서비스_업종_코드'].astype(str).str.startswith('CS1')&sa.q.str.endswith('2')].drop_duplicates(['q','행정동_코드','서비스_업종_코드']); sa['Y']=sa.q.str[:4].astype(int)
print(sa.groupby('Y')['당월_매출_금액'].sum().div(1e12).round(2).to_dict())
A=['연령대_20_매출_금액','연령대_30_매출_금액']; print([c for c in sa.columns if '연령대' in c][:7])
gs=sa.groupby(['행정동_코드','Y']); S=pd.DataFrame({'sales':gs['당월_매출_금액'].sum(),'tx':gs['당월_매출_건수'].sum(),'wk':gs['주말_매출_금액'].sum(),'y2030':gs[A[0]].sum()+gs[A[1]].sum()}).rename_axis(['adm','Y'])
S['lsales']=np.log1p(S.sales); S['ticket']=S.sales/S.tx; S['wkend']=S.wk/S.sales; S['s2030']=S.y2030/S.sales
S['s1']=S.sales/S.groupby('adm').sales.shift(1)-1
def q2(f):
    x=rcsv(RAW+f); x=x[x['기준_년분기_코드'].astype(str).str.endswith('2')].copy(); x['Y']=x['기준_년분기_코드'].astype(str).str[:4].astype(int); return x.set_index(['행정동_코드','Y'])
fl,wk,rp=q2('seoul_commercial_floating_population_dong.csv'),q2('seoul_workplace_population_dong.csv'),q2('seoul_resident_population_dong.csv')
E_=pd.DataFrame(index=fl.index); E_['lflow']=np.log1p(fl['총_유동인구_수']); E_['flow2030']=(fl['연령대_20_유동인구_수']+fl['연령대_30_유동인구_수'])/fl['총_유동인구_수']
E_['lwork']=np.log1p(wk['총_직장_인구_수']); E_['lres']=np.log1p(rp['총_상주인구_수']); E_['res2030']=(rp['연령대_20_상주인구_수']+rp['연령대_30_상주인구_수'])/rp['총_상주인구_수']; E_=E_.rename_axis(['adm','Y'])
q=p.join(S,how='inner').join(E_,how='left'); q['perstore']=np.log1p(q.sales/q.n); q=q[q.n>=100]
print('dong-years',len(q),q.groupby('Y').size().to_dict())
q=q.reset_index()
LIC=['logn','cafe_sh','g1','g3','open_r','close_r','near_g3']; SAL=['lsales','perstore','ticket','wkend','s2030','s1']; POP=['lflow','flow2030','lwork','lres','res2030']
sets={'점포':LIC,'+매출':LIC+SAL,'+인구':LIC+POP,'전체':LIC+SAL+POP}
def fit2(tr,te,F,kind):
    a=tr[F].median().fillna(0); X,Z=tr[F].fillna(a),te[F].fillna(a)
    m=HGB(max_depth=2,max_iter=80,learning_rate=.05,random_state=0) if kind=='h' else make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced',C=.3))
    return m.fit(X,tr.y).predict_proba(Z)[:,1]
for h in (2,1):
    t=q.copy(); ff=q.set_index(['adm','Y']).sales; idx=pd.MultiIndex.from_arrays([t.adm,t.Y+h]); t['fut']=ff.reindex(idx).values; t['fg']=t.fut/t.sales-1
    nf=q.set_index(['adm','Y']).n.reindex(idx).values; t['fn']=nf/t.n-1
    o=t[t.fg.notna()].copy(); o['y']=(o.groupby('Y').fg.rank(pct=True,ascending=False)<=.10).astype(int)
    print('\nh',h,'obs',o.groupby('Y').size().to_dict()); print('spearman 점포증가 vs 매출증가',{int(k):round(v[['fn','fg']].corr(method='spearman').iloc[0,1],3) for k,v in o.groupby('Y')})
    print('spearman 과거점포g3 vs 미래매출',{int(k):round(v[['g3','fg']].corr(method='spearman').iloc[0,1],3) for k,v in o.groupby('Y')}); print('spearman 과거매출s1 vs 미래매출',{int(k):round(v[['s1','fg']].corr(method='spearman').iloc[0,1],3) for k,v in o.groupby('Y') if v.s1.notna().sum()>50})
    tys=[T for T in sorted(o.Y.unique()) if (o.Y<=T-h).sum()>=200]
    rows=[]
    for T in tys:
        te=o[o.Y==T]; tr=o[o.Y<=T-h]; Y=te.y.values; g=te.fg.values; r={'T':T,'N':len(te),'trn':len(tr),'all_med':round(np.median(g),3)}
        def rec(nm,v):
            i=np.argsort(-v); r[nm]=f"{Y[i[:10]].sum()}/{Y[i[:20]].sum()} auc{roc_auc_score(Y,v):.2f} g{np.median(g[i[:10]]):+.3f}"
        rec('g3점포',te.g3.values); rec('g1점포',te.g1.values)
        if te.s1.notna().mean()>.5: rec('s1매출',te.s1.fillna(0).values)
        for nm,F in sets.items():
            for kd in 'lh': rec(f'{nm}_{kd}',fit2(tr,te,F,kd))
        rows.append(r)
    print(pd.DataFrame(rows).T.to_string())
# structure: 2024->2026 stores vs sales quadrants
t=q.copy(); idx=pd.MultiIndex.from_arrays([t.adm,t.Y+2]); t['fs']=q.set_index(['adm','Y']).sales.reindex(idx).values/t.sales-1; t['fn']=q.set_index(['adm','Y']).n.reindex(idx).values/t.n-1
for T in (2022,2023,2024):
    v=t[(t.Y==T)&t.fs.notna()&t.fn.notna()]; hi=v.fn.rank(pct=True)>=.9; print(T,'N',len(v),'top10% 점포증가 n',hi.sum(),'그중 매출증가율 중앙값이하',int((v.fs[hi]<v.fs.median()).sum()),'매출중앙 top',round(v.fs[hi].median(),3),'rest',round(v.fs[~hi].median(),3))
v=t[(t.Y==2024)]; v=v.assign(name=v.adm.map(names),gu2=v.adm.map(gu)); print(v[v.adm.isin(SS)|False][['name','n','sales','fn','fs']].round(3).to_string(index=False)) if False else None
SS={k:x for k,x in names.items() if str(x).startswith('성수')}; print(v[v.adm.isin(SS)][['name','n','fn','fs']].round(3).to_string(index=False))
v2=v[v.fn.notna()&v.fs.notna()]; hi=v2[v2.fn.rank(pct=True)>=.9].sort_values('fs'); print(hi[['gu2','name','n','fn','fs']].round(3).to_string(index=False))
