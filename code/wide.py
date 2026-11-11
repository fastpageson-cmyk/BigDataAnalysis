import sys,pandas as pd
O=sys.argv[1] if len(sys.argv)>1 else __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
F={'cafe':f'{O}/data/raw/인허가/휴게음식점_인허가_서울.csv','rest':f'{O}/data/raw/인허가/일반음식점_인허가_서울.csv'}
df=pd.concat([pd.read_csv(F[n],encoding='cp949',usecols=['관리번호','사업장명','인허가일자','폐업일자','영업상태명','업태구분명'],dtype=str,encoding_errors='replace').assign(src=n) for n in ['cafe','rest']]).drop_duplicates('관리번호')
df['open']=pd.to_datetime(df['인허가일자'].str.strip(),errors='coerce'); df['close']=pd.to_datetime(df['폐업일자'].str.strip(),errors='coerce')
df['nm']=df['사업장명'].fillna('').str.replace(r'\s+','',regex=True)
B={'커피':{'메가커피':r'메가(MGC|엠지씨)?커피','컴포즈':r'컴포즈커피','빽다방':r'빽다방','이디야':r'이디야','투썸':r'투썸플레이스','할리스':r'할리스','커피빈':r'커피빈','탐앤탐스':r'탐앤탐스','더벤티':r'더벤티','매머드':r'매머드(익스프레스|커피)','파스쿠찌':r'파스쿠찌','엔제리너스':r'엔제리너스','공차':r'^공차|공차[가-힣]*점$','요거프레소':r'요거프레소','스타벅스(직영)':r'스타벅스'},
'치킨':{'교촌':r'교촌','BBQ':r'BBQ|비비큐','bhc':r'(?i)bhc|비에이치씨','굽네':r'굽네','네네':r'네네치킨','처갓집':r'처갓집','호식이':r'호식이','페리카나':r'페리카나','멕시카나':r'멕시카나','푸라닭':r'푸라닭','노랑통닭':r'노랑통닭','60계':r'60계','자담':r'자담치킨','또래오래':r'또래오래','지코바':r'지코바','순살만공격':r'순살만공격'},
'버거·피자':{'맘스터치':r'맘스터치','롯데리아':r'롯데리아','도미노':r'도미노피자','피자헛':r'피자헛','미스터피자':r'미스터피자','파파존스':r'파파존스','피자스쿨':r'피자스쿨','반올림피자':r'반올림피자','피자알볼로':r'피자알볼로'},
'베이커리·디저트':{'파리바게뜨':r'파리바게','뚜레쥬르':r'뚜레쥬르','배스킨':r'배스킨|베스킨','설빙':r'설빙','던킨':r'던킨','와플대학':r'와플대학'},
'분식·한식':{'엽기떡볶이':r'엽기떡볶이','신전떡볶이':r'신전떡볶이','죠스떡볶이':r'죠스떡볶이','김밥천국':r'김밥천국','본죽':r'본죽','한솥':r'한솥','고봉민김밥':r'고봉민','바르다김선생':r'바르다김선생','이삭토스트':r'이삭토스트','써브웨이':r'써브웨이'},
'주점':{'역전할머니맥주':r'역전할머니','투다리':r'투다리','봉구비어':r'봉구비어','생활맥주':r'생활맥주'}}
R=[]
for g,d in B.items():
  for b,p in d.items():
    x=df[df.nm.str.contains(p,regex=True)]; op=x.open.dt.year; c=x[op.between(2018,2023)]
    R.append(dict(업종=g,브랜드=b,전체=len(x),영업중=x['영업상태명'].str.contains('영업',na=False).sum(),개업18_23=len(c),
      폐업2년내=int((c.close<=c.open+pd.DateOffset(years=2)).sum()),폐업률2년=round((c.close<=c.open+pd.DateOffset(years=2)).mean(),3) if len(c) else None,
      휴게비중=round((x.src=='cafe').mean(),2)))
r=pd.DataFrame(R); pd.set_option('display.width',250); print(r.to_string(index=False))
print('\n합계 개업18_23',r.개업18_23.sum(),'폐업2년내',r.폐업2년내.sum())
