# 서울시 지하철 호선별 역별 시간대별 승하차 인원(OA-12252, OpenAPI CardSubwayTime) 2015-01부터 월별 수집. 키는 .env에서 읽고 출력하지 않음
import os,json,time,urllib.request,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('SEOUL_OPENAPI_KEY_A=')][0]
rows=[]
for m in pd.period_range('2015-01','2026-10',freq='M'):
    ym=m.strftime('%Y%m'); s=1
    while True:
        u=f'http://openapi.seoul.go.kr:8088/{K}/json/CardSubwayTime/{s}/{s+999}/{ym}'
        for t in range(3):
            try: d=json.load(urllib.request.urlopen(u,timeout=60)); break
            except Exception as e: time.sleep(3); err=e
        else: raise SystemExit(f'{ym} 실패: {type(err).__name__}')
        b=d.get('CardSubwayTime')
        if not b: print(ym,'없음',d.get('RESULT',{}).get('CODE')); break
        rows+=b['row']; n=b['list_total_count']
        if s+999>=n: break
        s+=1000
    if b: print(ym,n,flush=True)
os.makedirs(f'{O}/data/raw/지하철',exist_ok=True)
df=pd.DataFrame(rows).drop_duplicates()  # 2026-07은 API가 같은 621행을 두 번 돌려줌(2026-11-11 확인)
df.to_csv(f'{O}/data/raw/지하철/subway_time_2015_2026.csv',index=False,encoding='utf-8-sig')
print(len(df),'rows',df.USE_MM.min(),df.USE_MM.max())
