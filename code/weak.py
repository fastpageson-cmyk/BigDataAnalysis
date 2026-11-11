import sys,pandas as pd,numpy as np
O=sys.argv[1] if len(sys.argv)>1 else __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
F={'cafe':f'{O}/data/raw/인허가/휴게음식점_인허가_서울.csv','rest':f'{O}/data/raw/인허가/일반음식점_인허가_서울.csv'}; SB='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv'
c=['관리번호','사업장명','인허가일자','폐업일자','영업상태명','업태구분명','지번주소','도로명주소','최종수정일자','좌표정보(X)']
df=pd.concat([pd.read_csv(F[n],encoding='cp949',usecols=c,dtype=str,encoding_errors='replace').assign(src=n) for n in ['cafe','rest']])
df['open']=pd.to_datetime(df['인허가일자'].str.strip(),errors='coerce'); df['close']=pd.to_datetime(df['폐업일자'].str.strip(),errors='coerce'); df['mod']=pd.to_datetime(df['최종수정일자'].str.strip(),errors='coerce')
B={'메가커피':r'메가\s*(?:MGC|엠지씨)?\s*커피','이디야':r'이디야','교촌':r'교촌','BBQ':r'BBQ|비비큐','bhc':r'bhc|BHC|비에이치씨'}
sb=pd.read_csv(SB,usecols=['상호명','상권업종소분류명'],low_memory=False)
for b,p in B.items():
    x=df[df['사업장명'].str.contains(p,regex=True,na=False)]; a=x[x['영업상태명'].str.contains('영업',na=False)]
    nm=x['사업장명'].str.replace(r'\s+','',regex=True)
    print(f'\n## {b}: 행 {len(x)} | 업태 상위',x['업태구분명'].value_counts().head(3).to_dict(),'| src',x.src.value_counts().to_dict())
    print(' 표기 종류',nm.str.extract('('+p+r'[^\s]{0,0})')[0].str.replace(r'\s','',regex=True).value_counts().head(5).to_dict())
    other=x[~x['업태구분명'].isin(['커피숍','까페','통닭(치킨)','호프/통닭','기타 휴게음식점','일반조리판매','패스트푸드','기타','분식'])]
    print(' 의심 업태 예',other[['사업장명','업태구분명']].head(3).values.tolist())
    k=a['지번주소'].fillna(a['도로명주소']).str.replace(r'\s+','',regex=True); print(' 영업중',len(a),'같은 주소 중복',k.duplicated().sum())
    print(' 영업중인데 허가 2015년 이전',(a.open.dt.year<2015).sum(),'| 폐업일자 없음 폐업',(x['영업상태명'].str.contains('폐업',na=False)&x.close.isna()).sum(),'| 휴업',(x['영업상태명'].str.contains('휴업',na=False)).sum())
    print(' 폐업연도',x.close.dt.year.value_counts().sort_index().loc[2015:].to_dict())
    print(' 상가정보 매칭 수',sb['상호명'].astype(str).str.contains(p,regex=True).sum())
ex=df[df['사업장명'].str.contains('교촌',na=False)&~df['사업장명'].str.contains('치킨|교촌F|교촌에프',na=False)]['사업장명'].head(6).tolist(); print('\n교촌 비치킨 예',ex)
ex=df[df['사업장명'].str.contains('BBQ|비비큐',na=False)&~df['사업장명'].str.contains('치킨|비비큐|BBQ치킨|bbq',case=False,na=False)]['사업장명'].head(6).tolist(); print('BBQ 기타 예',ex)
