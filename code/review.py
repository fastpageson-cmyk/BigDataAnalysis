# 제안서 재검토용 탐색: 브랜드 매칭 점포의 기간별 폐업률, 컬럼 채움률, 이전·직권말소 점검
import sys,os,re,pandas as pd,numpy as np
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
F={'cafe':f'{O}/data/raw/인허가/휴게음식점_인허가_서울.csv','rest':f'{O}/data/raw/인허가/일반음식점_인허가_서울.csv'}
C=['관리번호','사업장명','인허가일자','폐업일자','영업상태명','상세영업상태명','업태구분명','지번주소','좌표정보(X)','좌표정보(Y)','소재지면적','시설총규모','보증액','월세액','건물소유구분명','남성종사자수','여성종사자수','영업장주변구분명','다중이용업소여부','데이터갱신일자','최종수정일자']
df=pd.concat([pd.read_csv(F[n],encoding='cp949',usecols=C,dtype=str,encoding_errors='replace').assign(src=n) for n in F]).drop_duplicates('관리번호')
df.to_pickle(f'{O}/data/processed/_lic_all.pkl')

# 브랜드 사전은 wide.py의 B를 그대로 씀
src=open(f'{O}/code/wide.py',encoding='utf-8').read()
B=eval(src[src.index('B={')+2:src.index('\nR=[]')])
df['nm']=df['사업장명'].fillna('').str.replace(r'\s+','',regex=True)
df['open']=pd.to_datetime(df['인허가일자'].str.strip(),errors='coerce'); df['close']=pd.to_datetime(df['폐업일자'].str.strip(),errors='coerce')
rows=[]
for g,d in B.items():
    for b,p in d.items():
        m=df.nm.str.contains(p,regex=True)
        rows.append(df[m].assign(업종=g,브랜드=b))
s=pd.concat(rows)
s['n_match']=s.groupby('관리번호').브랜드.transform('size')
s.to_pickle(f'{O}/data/processed/_brand_stores.pkl')
print('브랜드 매칭',len(s),'중복매칭 관리번호',(s.n_match>1).sum())
