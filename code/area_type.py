# 상권 성격 프로필(2025 외식 추정매출의 요일·시간대·연령 구성) → 상권 유형 5개(KMeans)
import os,numpy as np,pandas as pd
from sklearn.cluster import KMeans
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d=pd.read_csv(f'{O}/data/raw/상권영역/상권_추정매출_2025.csv',dtype={'상권_코드':str})
f=d[d.서비스_업종_코드.str.startswith('CS1')]
g=f.groupby('상권_코드')[[c for c in f.columns if c.endswith('_매출_금액')]].sum()
P=pd.DataFrame({'외식매출(log)':np.log1p(g.당월_매출_금액/4),'주말비중':g.주말_매출_금액/g.당월_매출_금액,
  '점심비중':g['시간대_11~14_매출_금액']/g.당월_매출_금액,'야간비중':(g['시간대_21~24_매출_금액']+g['시간대_00~06_매출_금액'])/g.당월_매출_금액,
  '20대비중':g.연령대_20_매출_금액/g.당월_매출_금액,'30·40대비중':(g.연령대_30_매출_금액+g.연령대_40_매출_금액)/g.당월_매출_금액,
  '50대+비중':(g.연령대_50_매출_금액+g.연령대_60_이상_매출_금액)/g.당월_매출_금액,'여성비중':g.여성_매출_금액/g.당월_매출_금액}).replace([np.inf,-np.inf],np.nan).dropna()
# 업종별 매출 비중(커피·치킨 등 수요 신호)
cat={'커피':'커피-음료','치킨':'치킨전문점','버거·피자':'패스트푸드점','베이커리·디저트':'제과점','분식·한식':'분식전문점','주점':'호프-간이주점'}
for k,v in cat.items():
    P[f'{k}매출비중']=f[f.서비스_업종_코드_명==v].groupby('상권_코드').당월_매출_금액.sum().reindex(P.index).fillna(0)/g.당월_매출_금액.reindex(P.index)
X=P[['외식매출(log)','주말비중','점심비중','야간비중','20대비중','30·40대비중','50대+비중','여성비중']]; Z=(X-X.mean())/X.std()
km=KMeans(5,n_init=20,random_state=0).fit(Z); P['유형번호']=km.labels_
c=pd.DataFrame(km.cluster_centers_,columns=X.columns).round(2); c['상권수']=P.유형번호.value_counts().sort_index().values
print(c.to_string())
A=pd.read_pickle(f'{O}/data/processed/areas.pkl').set_index('상권코드')
for t in range(5): print(t, A.reindex(P.index[P.유형번호==t]).assign(m=P.외식매출(log) if False else P['외식매출(log)']).sort_values('m',ascending=False).상권명.head(8).tolist())
P.to_pickle(f'{O}/data/processed/area_profile.pkl')
