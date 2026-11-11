# 치킨 브랜드 비교표: 서울 인허가 3년 실제 폐업률(chicken_wide.py B_넓음) + 공정위 2024 전국 가맹점수·평균매출·창업비(마스킹 번호→브랜드명 매핑)
# 금액 단위: 천원(공정위 원자료). 창업비 = 가맹비+교육비+보증금+기타+인테리어(임차보증금·권리금 제외)
import os,subprocess,io,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util,sys,contextlib
spec=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(cw)
d=cw.D['B_넓음']; s=d[d.프랜차이즈].groupby('브랜드').agg(서울개업15_22=('y_실제','size'),서울3년실제폐업=('y_실제','mean'))
mp=pd.read_csv(f'{O}/data/processed/ftc_masked_brand_map.csv',dtype=str).set_index('alsBrandMngNo').brandNm
f=pd.read_csv(f'{O}/data/raw/ftc_eatout_2017_2025.csv',dtype=str); f['ftc']=f.alsBrandMngNo.map(mp); f=f[f.jngBizCrtrYr=='2024']
for x in ['frcsCnt','avgSlsAmt','jngBzmnJngAmt','jngBzmnEduAmt','jngBzmnGrnteAmt','jngBzmnEtcAmt','intrrAmt','storCrtrArea']: f[x]=pd.to_numeric(f[x],errors='coerce')
f['창업비']=f[['jngBzmnJngAmt','jngBzmnEduAmt','jngBzmnGrnteAmt','jngBzmnEtcAmt','intrrAmt']].sum(axis=1,min_count=1)
K={'교촌':'교촌치킨','BBQ':'BBQ','bhc':'비에이치씨(BHC)','굽네':'굽네치킨','푸라닭':'푸라닭','처갓집':'처갓집양념치킨','지코바':'지코바양념치킨','60계':'60계',
   '자담':'자담치킨','멕시카나':'멕시카나','호식이':'호식이두마리치킨','노랑통닭':'노랑통닭','깐부':'깐부치킨','네네':'네네치킨','페리카나':'페리카나','또래오래':'또래오래'}
g=f[f.ftc.isin(K.values())].set_index('ftc')[['frcsCnt','avgSlsAmt','창업비','storCrtrArea']].rename(columns={'frcsCnt':'전국가맹점24','avgSlsAmt':'평균매출24_천원','창업비':'창업비_천원','storCrtrArea':'기준면적_평'})
s['ftc']=s.index.map(K); t=s.join(g,on='ftc')
t['매출/창업비']=t.평균매출24_천원/t.창업비_천원
t=t.sort_values('서울3년실제폐업'); t.round(3).to_csv(f'{O}/data/processed/chicken_brand_table.csv',encoding='utf-8-sig')
pd.set_option('display.width',200); print(t.drop(columns='ftc').round(3).to_string())
