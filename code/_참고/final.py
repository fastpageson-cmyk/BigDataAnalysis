import sys,glob,pandas as pd,numpy as np
S=sys.argv[1]; R='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/'; RAW=R+'maximus-analysis-archive/data/raw/'; OUT=sys.argv[2]
pd.set_option('display.width',240,'display.max_columns',40,'display.max_rows',200)
def rc(f):
    for e in ['utf-8-sig','cp949']:
        try: return pd.read_csv(f,encoding=e,low_memory=False)
        except UnicodeDecodeError: pass
st=pd.concat([rc(f'{S}/stores/s{y}.csv') for y in (2021,2022)]+[rc(RAW+f'seoul_commercial_stores_dong_{y}.csv') for y in (2023,2024,2025)]+[rc(S+'/stores/sheet.csv')])
st['q']=st['기준_년분기_코드'].astype(str); st=st[st['서비스_업종_코드']=='CS100010'].sort_values('점포_수',na_position='first').drop_duplicates(['q','행정동_코드'],keep='last'); st['Y']=st.q.str[:4].astype(int)
print('stores dongs 2025',st[st.Y==2025]['행정동_코드'].nunique())
sa=pd.concat([rc(RAW+f'seoul_commercial_sales_dong_{y}.csv') for y in (2024,)]+[rc(S+'/data/sales.csv')]); sa['q']=sa['기준_년분기_코드'].astype(str); sa=sa[sa['서비스_업종_코드']=='CS100010'].drop_duplicates(['q','행정동_코드']); sa['Y']=sa.q.str[:4].astype(int)
fl=rc(RAW+'seoul_commercial_floating_population_dong.csv'); wk=rc(RAW+'seoul_workplace_population_dong.csv'); fl['Y']=fl['기준_년분기_코드'].astype(str).str[:4].astype(int); wk['Y']=wk['기준_년분기_코드'].astype(str).str[:4].astype(int)
def rate(y):
    x=st[st.Y==y].groupby('행정동_코드'); return x['폐업_점포_수'].sum(min_count=4)/x['점포_수'].mean()
d=pd.DataFrame({'폐업률_2023':rate(2023),'폐업률_2024':rate(2024),'폐업률_2025':rate(2025)})
d['폐업률_2년평균']=d[['폐업률_2024','폐업률_2025']].mean(axis=1,skipna=False); d['폐업률_직전2년']=d[['폐업률_2023','폐업률_2024']].mean(axis=1,skipna=False)
s25=st[st.q=='20254'].set_index('행정동_코드'); d['동']=s25['행정동_코드_명']; d['점포수_2025Q4']=s25['점포_수']; d['프랜차이즈비중']=(s25['프랜차이즈_점포_수']/s25['점포_수']).round(3)
d['개업률_2025']=st[st.Y==2025].groupby('행정동_코드')['개업_점포_수'].sum()/st[st.Y==2025].groupby('행정동_코드')['점포_수'].mean()
a=sa.groupby(['행정동_코드','Y'])[['당월_매출_금액','당월_매출_건수']].mean().unstack('Y')
d['분기평균매출_억_2025']=a[('당월_매출_금액',2025)]/1e8; d['분기평균건수_2025']=a[('당월_매출_건수',2025)]; d['매출_2024']=a[('당월_매출_금액',2024)]/1e8
d['매출증가율_2024_25']=d['분기평균매출_억_2025']/d['매출_2024']-1
d['유동_2025']=fl[fl.Y==2025].groupby('행정동_코드')['총_유동인구_수'].mean(); d['직장_2025']=wk[wk.Y==2025].groupby('행정동_코드')['총_직장_인구_수'].mean()
# 2026 상반기: 선정 뒤 데이터
h=st[st.q.isin(['20261','20262'])].groupby('행정동_코드')['폐업_점포_수'].sum(); d['폐업률_2026상반기연율']=h*2/d['점포수_2025Q4']
sq=sa.set_index(['행정동_코드','q'])['당월_매출_금액'].unstack('q'); d['매출변화_26Q2대비25Q2']=sq['20262']/sq['20252']-1
sb=pd.read_csv(R+'데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv',usecols=['행정동코드','시군구명','상권업종소분류명'],low_memory=False); gu=sb.drop_duplicates('행정동코드').set_index('행정동코드')['시군구명']; d['구']=d.index.map(gu)
print(sb[sb['상권업종소분류명'].astype(str).str.contains('카페|커피')]['상권업종소분류명'].value_counts().head(4).to_dict())
d['상가정보_카페수_2026_06']=sb[sb['상권업종소분류명'].astype(str).str.contains('카페')].groupby('행정동코드').size()
u=d[(d['점포수_2025Q4']>=20)&d['폐업률_2년평균'].notna()&d['분기평균매출_억_2025'].notna()].copy(); print('대상',len(u),'전체',len(d),'sbiz vs 공식 spearman',u[['점포수_2025Q4','상가정보_카페수_2026_06']].corr(method='spearman').iloc[0,1].round(3))
u['카페당_분기매출_만원']=u['분기평균매출_억_2025']*1e4/u['점포수_2025Q4']
pm,cm=u['분기평균매출_억_2025'].median(),u['폐업률_2년평균'].median(); print('중앙값: 매출억',round(pm,1),'폐업',round(cm,3))
ty=lambda p,c:np.where((p>=pm)&(c<cm),'검증형',np.where((p>=pm)&(c>=cm),'격전형',np.where((p<pm)&(c<cm),'숨은안정','위험형')))
u['유형']=ty(u['분기평균매출_억_2025'],u['폐업률_2년평균'])
pa=a[('당월_매출_금액',2024)].reindex(u.index)/1e8; pv,cv=pa,u['폐업률_직전2년']; u['직전유형']=np.where(cv.isna(),'-',ty(pv,cv)); u['유형유지']=u['유형']==u['직전유형']
# 유동 대비 결제 잔차
x=np.log(u['유동_2025']); y=np.log(u['분기평균건수_2025']); ok=x.notna()&y.notna(); c=np.polyfit(x[ok],y[ok],1); u['유동대비결제잔차']=y-(c[0]*x+c[1])
print(u.groupby('유형').agg(동수=('동','size'),유지=('유형유지','mean'),점포=('점포수_2025Q4','median'),매출억=('분기평균매출_억_2025','median'),카페당=('카페당_분기매출_만원','median'),폐업2년=('폐업률_2년평균','median'),폐업26=('폐업률_2026상반기연율','median'),매출26=('매출변화_26Q2대비25Q2','median'),프차=('프랜차이즈비중','median')).round(3).to_string())
perm=u['카페당_분기매출_만원'].median()
sel={'검증형':u[(u.유형=='검증형')&u.유형유지&(u['카페당_분기매출_만원']>=perm)].nsmallest(10,'폐업률_2년평균'),
 '격전형':u[u.유형=='격전형'].nlargest(10,'분기평균매출_억_2025'),
 '숨은안정':u[(u.유형=='숨은안정')&u.유형유지&(u['점포수_2025Q4']>=30)].nsmallest(10,'폐업률_2년평균'),
 '위험형':u[u.유형=='위험형'].nlargest(10,'폐업률_2년평균')}
cols=['구','동','점포수_2025Q4','분기평균매출_억_2025','카페당_분기매출_만원','폐업률_2년평균','폐업률_2026상반기연율','매출증가율_2024_25','매출변화_26Q2대비25Q2','유동대비결제잔차','프랜차이즈비중','유형유지']
for k,v in sel.items():
    print('\n',k,len(v),'| 2026폐업 중앙',round(v['폐업률_2026상반기연율'].median(),3)); print(v[cols].round(3).to_string(index=False))
print('\n성수'); print(u[u['동'].str.startswith('성수')][['동','유형','직전유형']+cols[2:7]].round(3).to_string(index=False))
print('anomaly: 카페당 매출 하위'); print(u.nsmallest(6,'카페당_분기매출_만원')[['구','동','유형','점포수_2025Q4','분기평균매출_억_2025','상가정보_카페수_2026_06']].round(2).to_string(index=False))
u=u.assign(선정=False); u.loc[pd.concat(sel.values()).index,'선정']=True
out=u.reset_index().rename(columns={'행정동_코드':'행정동코드'})[['유형','선정','구','동','행정동코드']+cols[2:]+['직전유형','폐업률_2023','폐업률_2024','폐업률_2025','개업률_2025','유동_2025','직장_2025','상가정보_카페수_2026_06']].sort_values(['유형','폐업률_2년평균']).round(4); out.to_csv(OUT,index=False,encoding='utf-8-sig'); print(len(out),OUT)
