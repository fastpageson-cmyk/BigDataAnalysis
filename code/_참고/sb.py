import sys,pandas as pd,numpy as np
O=sys.argv[1]; SB='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv'
pd.set_option('display.width',200)
s=pd.read_csv(SB,usecols=['상호명','지점명','상권업종소분류명','시군구명','행정동명','경도','위도'],low_memory=False)
for b,p in {'스타벅스':'스타벅스','메가커피':'메가.*커피|메가커피','올리브영':'올리브영','GS25':'GS25|지에스25','CU':'^CU|씨유','다이소':'다이소','교촌':'교촌'}.items():
    x=s[s['상호명'].astype(str).str.contains(p,regex=True)]
    print(b,len(x),'지점명있음',x['지점명'].notna().mean().round(2),'좌표',x['경도'].notna().mean().round(2),'업종',x['상권업종소분류명'].value_counts().head(2).to_dict())
df=pd.read_csv(f'{O}/data/cafe.csv',encoding='cp949',usecols=['관리번호','사업장명','인허가일자','폐업일자','영업상태명','좌표정보(X)','좌표정보(Y)','지번주소'],dtype=str,encoding_errors='replace').drop_duplicates('관리번호')
x=df[df['사업장명'].str.contains('스타벅스',na=False)].copy()
x['open']=pd.to_datetime(x['인허가일자'].str.strip(),errors='coerce'); x['close']=pd.to_datetime(x['폐업일자'].str.strip(),errors='coerce'); x['X']=pd.to_numeric(x['좌표정보(X)'],errors='coerce'); x['Y']=pd.to_numeric(x['좌표정보(Y)'],errors='coerce')
print('\n스타벅스 인허가',len(x),'좌표',x.X.notna().mean().round(3),'개업연도<1999',(x.open.dt.year<1999).sum())
print('개업연도',x.open.dt.year.value_counts().sort_index().loc[1999:].to_dict()); print('폐업연도',x.close.dt.year.value_counts().sort_index().to_dict())
print(x[x.open.dt.year<1999][['사업장명','인허가일자']].head(4).to_string(index=False))
a=x[x.close.isna()&x.X.notna()]; P=a[['X','Y']].values; D=np.sqrt(((P[:,None]-P[None])**2).sum(2)); np.fill_diagonal(D,1e9)
nn=D.min(1); print('\n영업중',len(a),'최근접 자사 거리 중앙값 m',int(np.median(nn)),'| 300m내 다른 지점 있음',(nn<300).mean().round(3),'500m',(nn<500).mean().round(3),'| 500m내 지점수 중앙',np.median((D<500).sum(1)))
# 폐업 지점: 폐업 시점 500m 내 영업 중 지점 수, 같은 시기 1km 내 신규 개업(이전 의심)
c=x[x.close.notna()&x.X.notna()&(x.close.dt.year>=2015)]
rel=0;nb=[]
for _,r in c.iterrows():
    d=np.sqrt((x.X-r.X)**2+(x.Y-r.Y)**2)
    act=(x.open<=r.close)&((x.close.isna())|(x.close>r.close))&(d<500)&(d>0); nb.append(act.sum())
    rel+=(((x.open-r.close).abs()<pd.Timedelta(days=180))&(d<1000)&(d>0)).any()
print('2015+ 폐업',len(c),'폐업 당시 500m내 자사 중앙',np.median(nb),'| 180일·1km 내 신규(이전 의심)',rel)
