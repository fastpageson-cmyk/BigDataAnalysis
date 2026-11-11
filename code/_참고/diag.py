import sys, pandas as pd, re
d=sys.argv[1]
cols=['관리번호','인허가일자','영업상태명','상세영업상태명','폐업일자','소재지면적','지번주소','도로명주소','업태구분명','좌표정보(X)','사업장명']
fr=[]
for n,src in [('cafe','휴게'),('rest','일반')]:
    x=pd.read_csv(f'{d}/{n}.csv',encoding='cp949',usecols=cols,dtype=str,encoding_errors='replace'); x['src']=src; fr.append(x)
df=pd.concat(fr); print(df.shape)
for c in ['인허가일자','폐업일자']:
    df[c]=pd.to_datetime(df[c].str.strip(),errors='coerce')
print(df.isna().mean().round(3))
print('dup id',df['관리번호'].duplicated().sum(),'full dup',df.duplicated().sum())
print(df.groupby('src')['영업상태명'].value_counts())
print('폐업 but no date',((df['영업상태명'].str.contains('폐업',na=False))&df['폐업일자'].isna()).sum())
print('close<open',(df['폐업일자']<df['인허가일자']).sum())
print(df['인허가일자'].dt.year.value_counts().sort_index().loc[1995:].to_string())
print(df['폐업일자'].dt.year.value_counts().sort_index().loc[1995:].to_string())
print(df.groupby('src')['업태구분명'].value_counts().groupby(level=0).head(14).to_string())
a=df['지번주소'].fillna('').str.extract(r'서울특별시\s+(\S+구)\s+(\S+?(?:동\d*가|동|가|로\d*가|로))(?:\s|$)')
print('addr parsed',a[0].notna().mean(), 'n dong',a.dropna().drop_duplicates().shape[0])
print(df.loc[a[0].isna(),['지번주소','도로명주소']].head(5).to_string())
print(a.dropna().assign(n=1).groupby([0,1]).n.sum().sort_values().head(8))
print(a[a[1].fillna('').str.startswith('성수')].value_counts())
