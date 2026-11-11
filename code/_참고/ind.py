import sys,pandas as pd,numpy as np
from scipy.stats import spearmanr
O=sys.argv[1]; RAW='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/maximus-analysis-archive/data/raw/'
pd.set_option('display.width',220,'display.max_rows',80)
def rc(f):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,low_memory=False)
        except UnicodeDecodeError: pass
st=pd.concat([rc(f'{O}/stores/s{y}.csv') for y in (2021,2022)]+[rc(RAW+f'seoul_commercial_stores_dong_{y}.csv') for y in (2023,2024,2025)])
st=st.drop_duplicates(['기준_년분기_코드','행정동_코드','서비스_업종_코드']); st['Y']=st['기준_년분기_코드'].astype(str).str[:4].astype(int)
k=['행정동_코드','서비스_업종_코드','Y']; g=st.groupby(k)
P=pd.DataFrame({'clo':g['폐업_점포_수'].sum(min_count=4)/g['점포_수'].mean(),'n':g['점포_수'].mean(),'fr':g['프랜차이즈_점포_수'].mean()/g['점포_수'].mean()})
names=st.drop_duplicates('서비스_업종_코드').set_index('서비스_업종_코드')['서비스_업종_코드_명']
print('업종 수',st['서비스_업종_코드'].nunique(),'| 동×업종×연 셀',len(P))
sa=pd.concat([rc(RAW+f'seoul_commercial_sales_dong_{y}.csv') for y in (2023,2024)]+[rc(O+'/data/sales.csv')]+[rc(f) for f in __import__('glob').glob(O+'/sales/x*/*.csv')])
sa=sa.drop_duplicates(['기준_년분기_코드','행정동_코드','서비스_업종_코드']); sa['Y']=sa['기준_년분기_코드'].astype(str).str[:4].astype(int)
Sv=sa.groupby(k)['당월_매출_금액'].mean().rename('sales'); P=P.join(Sv)
w=P.clo.unstack('Y')
print('셀 자기상관(점포10+):',{f'{a}->{b}':round(spearmanr(w.loc[P.n.unstack('Y')[a]>=10,a],w.loc[P.n.unstack('Y')[a]>=10,b],nan_policy='omit').statistic,3) for a,b in ((2022,2023),(2023,2024),(2024,2025))})
res=[];cells=[]
for y0 in (2023,2024):
    a=P.xs(y0,level='Y'); b=P.xs(y0-1,level='Y'); c=P.xs(y0+1,level='Y')
    t=pd.DataFrame({'clo':(a.clo+b.clo)/2,'n':a.n,'sales':a.sales,'next':c.clo,'fr':a.fr}).dropna(); t=t[t.n>=10]
    t['pp']=t.groupby(level=1).sales.rank(pct=True); t['cp']=t.groupby(level=1).clo.rank(pct=True)
    t['quad']=np.select([(t.pp>=.5)&(t.cp<.5),(t.pp>=.5)&(t.cp>=.5),(t.pp<.5)&(t.cp<.5)],['검증형','격전형','숨은안정'],'위험형')
    t['nr']=t.groupby(level=1)['next'].rank(pct=True); t['y']=t.nr>=.8
    for q_,v in t.groupby('quad'): res.append(dict(입력=f'{y0-1}-{y0}',유형=q_,셀=len(v),다음해폐업=round(v['next'].median(),3),위험상위20=round(v.y.mean(),3)))
    # 업종 간: 격전형 비중 높은 업종
    cells.append(t.assign(yr=y0))
R=pd.DataFrame(res); print(R.pivot(index='유형',columns='입력',values=['셀','다음해폐업','위험상위20']).to_string())
T=pd.concat(cells); ind=T.groupby(level=1).agg(셀=('clo','size'),폐업=('clo','median'),다음=('next','median'),격전비=('quad',lambda s:(s=='격전형').mean())).query('셀>=60').sort_values('다음',ascending=False)
ind.index=ind.index.map(names); print(ind.round(3).head(12).to_string()); print(ind.round(3).tail(6).to_string()); print('업종수(셀60+)',len(ind))
# baseline vs 2yr: predict next-year top20% closures within 업종
from sklearn.metrics import roc_auc_score
for y0 in (2023,2024):
    t=T[T.yr==y0]; print(y0,'AUC 2년평균폐업',round(roc_auc_score(t.y,t.clo),3),'올해만',round(roc_auc_score(t.y,P.xs(y0,level='Y').clo.reindex(t.index).fillna(0)),3),'매출',round(roc_auc_score(t.y,-t.sales.rank()),3),'프랜차이즈비',round(roc_auc_score(t.y,t.fr),3))
