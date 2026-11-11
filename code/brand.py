import sys,re,pandas as pd
O=sys.argv[1] if len(sys.argv)>1 else __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
F={'cafe':f'{O}/data/raw/인허가/휴게음식점_인허가_서울.csv','rest':f'{O}/data/raw/인허가/일반음식점_인허가_서울.csv'}
df=pd.concat([pd.read_csv(F[n],encoding='cp949',usecols=['관리번호','사업장명','인허가일자','폐업일자','영업상태명','업태구분명','지번주소'],dtype=str,encoding_errors='replace') for n in ['cafe','rest']]).drop_duplicates('관리번호')
df['open']=pd.to_datetime(df['인허가일자'].str.strip(),errors='coerce'); df['close']=pd.to_datetime(df['폐업일자'].str.strip(),errors='coerce')
B={'메가커피':r'메가\s*(MGC|엠지씨)?\s*커피|메가커피','컴포즈':r'컴포즈\s*커피','빽다방':r'빽다방','이디야':r'이디야','스타벅스':r'스타벅스','투썸':r'투썸\s*플레이스','교촌':r'교촌','BBQ':r'BBQ|비비큐','bhc':r'bhc|BHC|비에이치씨','굽네':r'굽네','맘스터치':r'맘스터치','파리바게뜨':r'파리바게','배스킨':r'배스킨|베스킨','설빙':r'설빙'}
rows=[]
for b,p in B.items():
    x=df[df['사업장명'].str.contains(p,regex=True,na=False)]
    op=x.open.dt.year
    rows.append(dict(브랜드=b,전체=len(x),영업중=(x['영업상태명'].str.contains('영업',na=False)).sum(),폐업=x.close.notna().sum(),첫개업=int(op.min()) if len(x) else None,개업_2015_19=op.between(2015,2019).sum(),개업_2020_23=op.between(2020,2023).sum(),개업_2024_26=op.between(2024,2026).sum(),
        개업18_23_2년생존=round(((x.close.isna())|(x.close>x.open+pd.DateOffset(years=2)))[op.between(2018,2023)].mean(),3)))
print(pd.DataFrame(rows).to_string(index=False))
print(df[df['사업장명'].str.contains(B['메가커피'],na=False)]['사업장명'].value_counts().head(5).to_dict())
