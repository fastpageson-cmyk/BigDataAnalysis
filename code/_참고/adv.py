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
    e=pd.Timestamp(f'{Y}-09-30'); s=e-pd.DateOffset(years=1)
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
MIN=100
def mk(h):
    t=p.copy(); t['fg']=p.n.groupby('adm').shift(-h)/p.n-1
    t=t[(t.n>=MIN)&t.g3.notna()].reset_index(); o=t[t.fg.notna()].copy(); o['y']=(o.groupby('Y')['fg'].rank(pct=True,ascending=False)<=.10).astype(int); return t,o
def fit(tr,te,F,kind='hgb'):
    a=tr[F].median(); X,Z=tr[F].fillna(a),te[F].fillna(a)
    mdl=HGB(max_depth=3,max_iter=150,learning_rate=.05,random_state=0) if kind=='hgb' else make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced'))
    return mdl.fit(X,tr.y).predict_proba(Z)[:,1]
t,o=mk(3); print('LONG rows',len(o),'dongs',o.adm.nunique())
res=[];rs=[]
SS={k:v for k,v in names.items() if str(v).startswith('성수')}; print(SS)
for T in range(2011,2024):
    tr=o[o.Y<=T-3]; te=o[o.Y==T]
    if len(tr)<300 or te.empty: continue
    ph,pl=fit(tr,te,BASE),fit(tr,te,BASE,'l'); y=te.y.values
    row=dict(T=T,N=len(te))
    for nm,v in dict(hgb=ph,logit=pl,g3=te.g3.values,g1=te.g1.values).items(): row[nm]=y[np.argsort(-v)[:10]].sum(); row[nm+'_auc']=roc_auc_score(y,v)
    res.append(row)
    tt=te.assign(p=ph); tt['r']=tt.p.rank(ascending=False); tt['rg']=tt.g3.rank(ascending=False); tt['rr']=tt.fg.rank(ascending=False)
    for _,q in tt[tt.adm.isin(SS)].iterrows(): rs.append(dict(T=T,dong=SS[q.adm],N=len(tt),n=int(q.n),model=int(q.r),mom=int(q.rg),real=int(q.rr)))
R1=pd.DataFrame(res); print(R1.round(2).to_string(index=False)); print(R1.mean(numeric_only=True).round(2).to_dict())
S=pd.DataFrame(rs); print(S.pivot(index='T',columns='dong',values=['model','mom','real']).to_string())
# recent model with DAT dong features, h=2
def rd(f,cols):
    x=pd.read_csv(RAW+f,encoding='utf-8-sig',low_memory=False); x=x[x['기준_년분기_코드'].astype(str).str.endswith('3')].copy(); x['Y']=x['기준_년분기_코드'].astype(str).str[:4].astype(int); return x.set_index(['행정동_코드','Y'])[cols]
def rdany(f):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(RAW+f,encoding=e,low_memory=False)
        except UnicodeDecodeError: pass
fl=rdany('seoul_commercial_floating_population_dong.csv'); print([c for c in fl.columns][12:25])
wk=rdany('seoul_workplace_population_dong.csv'); rp=rdany('seoul_resident_population_dong.csv'); fc=rdany('seoul_attracting_facilities_dong.csv'); print(list(fc.columns[3:]))
def q3(x):
    x=x[x['기준_년분기_코드'].astype(str).str.endswith('3')].copy(); x['Y']=x['기준_년분기_코드'].astype(str).str[:4].astype(int); return x.set_index(['행정동_코드','Y'])
fl,wk,rp,fc=q3(fl),q3(wk),q3(rp),q3(fc)
E_=pd.DataFrame(index=fl.index)
E_['lflow']=np.log1p(fl['총_유동인구_수']); E_['flow2030']=(fl['연령대_20_유동인구_수']+fl['연령대_30_유동인구_수'])/fl['총_유동인구_수']
E_['flow_f']=fl['여성_유동인구_수']/fl['총_유동인구_수']
E_['lwork']=np.log1p(wk['총_직장_인구_수']); E_['lres']=np.log1p(rp['총_상주인구_수']); E_['res2030']=(rp['연령대_20_상주인구_수']+rp['연령대_30_상주인구_수'])/rp['총_상주인구_수']
E_['lfac']=np.log1p(fc['집객시설_수']); 
tr_cols=[c for c in fc.columns if ('지하철' in c or '버스' in c or '극장' in c or '숙박' in c or '대학' in c) and '수' in c]
for c in tr_cols: E_['f_'+c]=fc[c].fillna(0)
E_=E_.rename_axis(['adm','Y'])
G=E_.groupby('adm'); E_['dflow']=E_['lflow']-G['lflow'].shift(1); E_['dwork']=E_['lwork']-G['lwork'].shift(1)
FLOW=['lflow','flow2030','flow_f','dflow']; WR=['lwork','lres','res2030','dwork']; FAC=['lfac']+['f_'+c for c in tr_cols]
t2,o2=mk(2); o2=o2.merge(E_.reset_index(),on=['adm','Y'],how='left'); print('RECENT avail',o2.groupby('Y').lflow.apply(lambda s:s.notna().sum()).loc[2020:].to_dict())
sets={'인허가만':BASE,'+유동':BASE+FLOW,'+직장·상주':BASE+WR,'+집객·교통':BASE+FAC,'전체':BASE+FLOW+WR+FAC}
out=[]
for T,trY in [(2023,[2021]),(2024,[2021,2022])]:
    te=o2[o2.Y==T]; tr=o2[o2.Y.isin(trY)]; y=te.y.values; row={'T':T,'N':len(te),'pos':y.sum(),'trn':len(tr)}
    for k in ['g1','g3']: v=te[k].values; row[k+'@10']=y[np.argsort(-v)[:10]].sum(); row[k+'@20']=y[np.argsort(-v)[:20]].sum(); row[k+'_auc']=round(roc_auc_score(y,v),3)
    for nm,F in sets.items():
        for kd in ['hgb','l']:
            v=fit(tr,te,F,kd); row[f'{nm}/{kd}@10']=y[np.argsort(-v)[:10]].sum(); row[f'{nm}/{kd}@20']=y[np.argsort(-v)[:20]].sum(); row[f'{nm}/{kd}_auc']=round(roc_auc_score(y,v),3)
    out.append(row)
print(pd.DataFrame(out).T.to_string())
# long-trained model plus recent: does long train beat short for same folds?
for T in [2023,2024]:
    te=o2[o2.Y==T]; y=te.y.values
    tr=o2[o2.Y<=T-2]; v=fit(tr,te,BASE); print('long-train base h2',T,len(tr),'@10',y[np.argsort(-v)[:10]].sum(),'@20',y[np.argsort(-v)[:20]].sum(),'auc',round(roc_auc_score(y,v),3))
# sales validation
ss=[]
for yy in [2023,2024,2025]:
    x=rdany(f'seoul_commercial_sales_dong_{yy}.csv'); x=x[x['기준_년분기_코드'].astype(str).str.endswith('3')&x['서비스_업종_코드'].astype(str).str.startswith('CS1')]
    ss.append(x.groupby('행정동_코드')['당월_매출_금액'].sum().rename(yy))
sl=pd.concat(ss,axis=1); st=[]
x=rdany('seoul_commercial_stores_dong_2023.csv'); x=x[x['기준_년분기_코드'].astype(str).str.endswith('3')&x['서비스_업종_코드'].astype(str).str.startswith('CS1')]; offi=x.groupby('행정동_코드')['점포_수'].sum()
c=pd.concat([p.xs(2023,level='Y').n.rename('lic'),offi.rename('off')],axis=1).dropna(); print('official vs license 2023 corr',c.corr(method='spearman').iloc[0,1].round(3),'ratio med',(c.lic/c.off).median().round(2),len(c))
v=pd.concat([(p.xs(2025,level='Y').n/p.xs(2023,level='Y').n-1).rename('sg'),(sl[2025]/sl[2023]-1).rename('rg'),p.xs(2023,level='Y').n.rename('n')],axis=1).dropna(); v=v[v.n>=MIN]
print('N',len(v),'spearman store vs sales growth',v[['sg','rg']].corr(method='spearman').iloc[0,1].round(3)); top=v.sg.rank(pct=True,ascending=False)<=.10
print('sales growth median top10% store-growth',v[top].rg.median().round(3),'others',v[~top].rg.median().round(3))
print('성수 sales/stores'); print(v[v.index.isin(SS)].assign(dong=lambda d:d.index.map(SS)).round(3).to_string())
# latest candidates: T=2024 train incl 2022,2024? use T=2026 base features with long model h=3
te=t[t.Y==2026].copy(); te['p']=fit(o,te,BASE); te['name']=te.adm.map(names); te['g']=te.gu
print(te.sort_values('p',ascending=False).head(12)[['g','name','n','g3','near_g3','p']].round(2).to_string(index=False))
print('성수 2026'); print(te[te.adm.isin(SS)][['name','n','g1','g3','close_r','open_r']].round(3).to_string(index=False))
