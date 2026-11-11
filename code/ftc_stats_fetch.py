# 공정위 가맹정보 브랜드별 가맹점 현황(data.go.kr 15110241) 연도별 수집: 브랜드명·가맹점수·신규·계약종료·해지·명의변경·평균매출. 키는 .env에서 읽고 출력하지 않음
import os,json,time,urllib.parse,urllib.request,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('DATA_GO_KR_KEY2=')][0]
U='https://apis.data.go.kr/1130000/FftcBrandFrcsStatsService/getBrandFrcsStats'
rows=[]
for yr in range(2017,2026):
    p=1
    while True:
        q=urllib.parse.urlencode(dict(serviceKey=K,pageNo=p,numOfRows=1000,resultType='json',yr=yr))
        for t in range(3):
            try: d=json.load(urllib.request.urlopen(U+'?'+q,timeout=60)); break
            except Exception as e: time.sleep(3); err=e
        else: raise SystemExit(f'{yr} p{p} 실패: {type(err).__name__}')
        it=d.get('items') or []
        if isinstance(it,dict): it=it.get('item',[])
        rows+=it; tot=int(d.get('totalCount',0) or 0)
        if p==1: print(yr,'total',tot,flush=True)
        if p*1000>=tot or not it: break
        p+=1
df=pd.DataFrame(rows).drop(columns=['jnghdqrtrsRprsvNm'],errors='ignore')  # 대표자 이름(개인정보) 저장 안 함
df.to_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv',index=False,encoding='utf-8-sig')
print(len(df),'rows', list(df.columns))
