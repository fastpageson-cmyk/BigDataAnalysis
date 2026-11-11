import sys, numpy as pd_np, pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
d=sys.argv[1]
cols=['관리번호','인허가일자','영업상태명','폐업일자','소재지면적','지번주소','업태구분명','좌표정보(X)','좌표정보(Y)']
df=pd.concat([pd.read_csv(f'{d}/{n}.csv',encoding='cp949',usecols=cols,dtype=str,encoding_errors='replace') for n in ['cafe','rest']])
n0=len(df); df=df.drop_duplicates(); n1=len(df); df=df.drop_duplicates('관리번호',keep='last'); n2=len(df)
for c in ['인허가일자','폐업일자']: df[c]=pd.to_datetime(df[c].str.strip(),errors='coerce')
a=df['지번주소'].fillna('').str.extract(r'서울특별시\s+(\S+구)\s+(\S+?(?:동\d*가|동|가|로\d*가|로))(?:\s|$)')
df['gu'],df['dong']=a[0],a[1]; df=df[df.gu.notna()]; n3=len(df)
closed=df['영업상태명'].str.contains('폐업',na=False)
df=df[~(closed&df['폐업일자'].isna())]; n4=len(df)
bad={'편의점','백화점','푸드트럭','철도역구내','키즈카페','기타 휴게음식점'}
df=df[~df['업태구분명'].isin(bad)]; n5=len(df)
print('rows',n0,'->dedup',n1,'->id',n2,'->addr',n3,'->closedate',n4,'->type',n5)
df['cafe']=df['업태구분명'].isin(['커피숍','까페','떡카페','전통찻집'])
df['west']=df['업태구분명'].isin(['경양식','외국음식전문점(인도,태국등)'])
df['area']=pd.to_numeric(df['소재지면적'],errors='coerce'); df['x']=pd.to_numeric(df['좌표정보(X)'],errors='coerce'); df['y_']=pd.to_numeric(df['좌표정보(Y)'],errors='coerce')
df['k']=df.gu+' '+df.dong
op=df['인허가일자']; cl=df['폐업일자'].fillna(pd.Timestamp('2099-1-1'))
recs=[]
for Y in range(2002,2026):
    e=pd.Timestamp(f'{Y}-12-31'); s=pd.Timestamp(f'{Y}-01-01')
    act=(op<=e)&(cl>e)
    g=df[act].groupby('k')
    r=pd.DataFrame({'n':g.size(),'cafe':g['cafe'].sum(),'west':g['west'].sum(),'area':g['area'].median()})
    r['opens']=df[(op>=s)&(op<=e)].groupby('k').size(); r['closes']=df[(cl>=s)&(cl<=e)].groupby('k').size()
    r['Y']=Y; recs.append(r.fillna({'opens':0,'closes':0}).reset_index())
p=pd.concat(recs); p['gu']=p.k.str.split().str[0]
xy=df[df.x>0].groupby('k')[['x','y_']].median()
s=p.set_index(['k','Y']).sort_index()
def lag(col,h): return s[col].groupby('k').shift(h)
s['g1']=s.n/lag('n',1)-1; s['g3']=s.n/lag('n',3)-1; s['cg3']=(s.cafe+1)/(lag('cafe',3)+1)-1
s['cafe_sh']=s.cafe/s.n; s['west_sh']=s.west/s.n; s['open_r']=s.opens/s.n; s['close_r']=s.closes/s.n; s['logn']=np.log(s.n); s['logcafe']=np.log1p(s.cafe)
s['gu_g3']=s.groupby(['gu','Y'])['g3'].transform('median')
# neighbours within 1.5km, count-weighted growth
ks=xy.index.to_numpy(); P=xy.to_numpy(); D=np.sqrt(((P[:,None]-P[None])**2).sum(2))
nb={k:ks[(D[i]<1500)&(D[i]>0)] for i,k in enumerate(ks)}
wide=s.n.unstack('Y')
ng={}
for y in range(2005,2026):
    v={}
    for k,ne in nb.items():
        ne=[q for q in ne if q in wide.index]
        if ne:
            a_,b_=wide.loc[ne,y].sum(),wide.loc[ne,y-3].sum(); v[k]=a_/b_-1 if b_>0 else np.nan
    ng[y]=pd.Series(v)
s['near_g3']=pd.DataFrame(ng).rename_axis(columns='Y').stack().reindex(s.index)
FE=['logn','logcafe','cafe_sh','west_sh','g1','g3','cg3','open_r','close_r','area','gu_g3','near_g3']
MIN=30
def build(h=3,target='n'):
    f=s[target if target=='n' else 'n'].copy()
    t=s.copy(); fut=s[target].groupby('k').shift(-h); base=s[target]
    t['fg']=(fut+(0 if target=='n' else 1))/(base+(0 if target=='n' else 1))-1
    t=t[(t.n>=MIN)&t.g3.notna()].reset_index()
    o=t[t.fg.notna()].copy()
    o['y']=(o.groupby('Y')['fg'].rank(pct=True,ascending=False)<=.10).astype(int)
    return t,o
def ev(o,h,years,models=True,shuffle=False):
    res=[]; rng=np.random.default_rng(0)
    for T in years:
        tr=o[o.Y<=T-h]; te=o[o.Y==T]
        if len(tr)<300 or len(te)==0: continue
        Xtr,Xte=tr[FE].fillna(tr[FE].median()),te[FE].fillna(tr[FE].median())
        ytr=tr.y.values
        if shuffle: ytr=rng.permutation(ytr)
        row={'T':T,'n':len(te),'pos':te.y.sum()}
        ps={'logit':make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')).fit(Xtr,ytr).predict_proba(Xte)[:,1],
            'hgb':HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=.05,random_state=0).fit(Xtr,ytr).predict_proba(Xte)[:,1],
            'g3':te.g3.values,'g1':te.g1.values,'cg3':te.cg3.values}
        for m,v in ps.items():
            row[m+'_p10']=te.y.values[np.argsort(-v)[:10]].sum(); row[m+'_auc']=roc_auc_score(te.y,v)
        res.append(row)
    return pd.DataFrame(res)
pd.set_option('display.width',220,'display.max_columns',40)
t,o=build(); print('panel rows',len(o),'dongs',o.k.nunique(),'pos rate',o.y.mean().round(3))
yrs=range(2011,2023)
R=ev(o,3,yrs); print(R.round(2).to_string()); M=R.mean(numeric_only=True).round(3); print(M.to_string())
print('expected random p10 ≈',(R.pos/R.n*10).mean().round(2))
print('PRE-COVID T<=2016',R[R['T']<=2016].mean(numeric_only=True).round(2)[['logit_p10','hgb_p10','g3_p10','g1_p10','cg3_p10','logit_auc','hgb_auc','g3_auc']].to_dict())
print('COVID T>=2017',R[R['T']>=2017].mean(numeric_only=True).round(2)[['logit_p10','hgb_p10','g3_p10','g1_p10','cg3_p10','logit_auc','hgb_auc','g3_auc']].to_dict())
Rs=ev(o,3,yrs,shuffle=True); print('shuffled labels',Rs.mean(numeric_only=True).round(2)[['logit_p10','hgb_p10','logit_auc','hgb_auc']].to_dict())
for h,tg,lb in [(2,'n','h2'),(4,'n','h4'),(3,'cafe','cafe y')]:
    tt,oo=build(h,tg); rr=ev(oo,h,range(2011,2026-h)); print(lb,rr.mean(numeric_only=True).round(2)[['logit_p10','hgb_p10','g3_p10','g1_p10','logit_auc','hgb_auc','g3_auc']].to_dict())
# Seongsu series and rank from out-of-time model
print(s.loc[['성동구 성수동1가','성동구 성수동2가'],['n','cafe','g3','cg3']].unstack(0).loc[2008:2025].round(2).to_string() if False else '')
for k in ['성동구 성수동1가','성동구 성수동2가']:
    print(k); print(s.loc[k].loc[2008:2025,['n','cafe','opens','closes']].T.astype(int).to_string())
rows=[]
for T in yrs:
    tr=o[o.Y<=T-3]; te=t[t.Y==T].copy()
    if len(tr)<300: continue
    m=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')).fit(tr[FE].fillna(tr[FE].median()),tr.y)
    te['p']=m.predict_proba(te[FE].fillna(tr[FE].median()))[:,1]; te['r']=te.p.rank(ascending=False); te['rg']=te.g3.rank(ascending=False); te['fr']=te.fg.rank(ascending=False)
    for _,q in te[te.k.isin(['성동구 성수동1가','성동구 성수동2가'])].iterrows(): rows.append(dict(T=T,k=q.k[4:],N=len(te),model=int(q.r),g3rank=int(q.rg),fg=None if pd.isna(q.fg) else round(q.fg,2),realrank=None if pd.isna(q.fr) else int(q.fr)))
print(pd.DataFrame(rows).to_string())
# coefficients on all data through 2022, and forecast 2025
tr=o; sc=StandardScaler().fit(tr[FE].fillna(tr[FE].median())); lr=LogisticRegression(max_iter=2000,class_weight='balanced').fit(sc.transform(tr[FE].fillna(tr[FE].median())),tr.y)
print(pd.Series(lr.coef_[0],FE).sort_values().round(2).to_string())
te=t[t.Y==2025].copy(); te['p']=lr.predict_proba(sc.transform(te[FE].fillna(tr[FE].median())))[:,1]
print(te.sort_values('p',ascending=False).head(12)[['k','n','cafe','g3','near_g3','p']].round(2).to_string(index=False))

print('\n=== ROBUSTNESS ===')
def build2(minbase,kind,h=3):
    t=s.copy(); b=s.n; f=s.n.groupby('k').shift(-h); fc=s.cafe.groupby('k').shift(-h)
    t['fg']={'rate':f/b-1,'abs':f-b,'cafeabs':fc-s.cafe}[kind]
    t=t[(t.n>=minbase)&t.g3.notna()].reset_index(); o=t[t.fg.notna()].copy()
    o['y']=(o.groupby('Y')['fg'].rank(pct=True,ascending=False)<=.10).astype(int); return t,o
def ev2(o,years,h=3):
    res=[]
    for T in years:
        tr=o[o.Y<=T-h]; te=o[o.Y==T]
        if len(tr)<300 or len(te)==0: continue
        Xtr,Xte=tr[FE].fillna(tr[FE].median()),te[FE].fillna(tr[FE].median())
        hg=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=.05,random_state=0).fit(Xtr,tr.y).predict_proba(Xte)[:,1]
        lr_=make_pipeline(StandardScaler(),LogisticRegression(max_iter=2000,class_weight='balanced')).fit(Xtr,tr.y).predict_proba(Xte)[:,1]
        abs3=(te.n-te.n/(1+te.g3)).values; cabs3=(te.cafe-((te.cafe+1)/(1+te.cg3)-1)).values
        vs={'hgb':hg,'logit':lr_,'g3':te.g3.values,'g1':te.g1.values,'abs3':abs3,'cafeabs3':cabs3,'n':te.n.values}
        row={'T':T,'N':len(te),'rand':te.y.mean()*10}
        for m,v in vs.items(): row[m]=te.y.values[np.argsort(-v)[:10]].sum()
        res.append(row)
    return pd.DataFrame(res)
for mb,kd in [(100,'rate'),(100,'abs'),(100,'cafeabs'),(30,'abs')]:
    t2,o2=build2(mb,kd); r=ev2(o2,range(2011,2023)); print(mb,kd,'N',round(r.N.mean()),r.mean(numeric_only=True).drop(['T','N']).round(2).to_dict(),'| pre2017',r[r['T']<=2016].mean(numeric_only=True).drop(['T','N']).round(2).to_dict())
t2,o2=build2(100,'rate')
rows=[]
for T in range(2011,2023):
    tr=o2[o2.Y<=T-3]; te=t2[t2.Y==T].copy()
    if len(tr)<300: continue
    m=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=.05,random_state=0).fit(tr[FE].fillna(tr[FE].median()),tr.y)
    te['p']=m.predict_proba(te[FE].fillna(tr[FE].median()))[:,1]; te['r']=te.p.rank(ascending=False); te['rg']=te.g3.rank(ascending=False); te['fr']=te.fg.rank(ascending=False)
    for _,q in te[te.k=='성동구 성수동1가'].iterrows(): rows.append(dict(T=T,N=len(te),model=int(q.r),mom=int(q.rg),real=int(q.fr)))
print(pd.DataFrame(rows).to_string(index=False))
m=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=.05,random_state=0).fit(o2[FE].fillna(o2[FE].median()),o2.y)
te=t2[t2.Y==2025].copy(); te['p']=m.predict_proba(te[FE].fillna(o2[FE].median()))[:,1]
print(len(te)); print(te.sort_values('p',ascending=False).head(10)[['k','n','cafe','g3','near_g3','p']].round(2).to_string(index=False))
print('hits among T=2014 top10:'); 
tr=o2[o2.Y<=2011]; te=o2[o2.Y==2014].copy(); mm=HistGradientBoostingClassifier(max_depth=3,max_iter=150,learning_rate=.05,random_state=0).fit(tr[FE].fillna(tr[FE].median()),tr.y)
te['p']=mm.predict_proba(te[FE].fillna(tr[FE].median()))[:,1]; print(te.sort_values('p',ascending=False).head(10)[['k','n','fg','y']].round(2).to_string(index=False))
