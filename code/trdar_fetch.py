# 서울시 상권분석서비스 상권 단위(1,650개) 분기 자료 2021Q1~ 수집, 치킨전문점(CS100007)만 저장. 키는 .env에서 읽고 출력하지 않음
#  추정매출-상권(OA-15572, VwsmTrdarSelngQq) → data/raw/상권_분기/치킨_매출_상권.csv
#  점포-상권(OA-15577, VwsmTrdarStorQq)     → data/raw/상권_분기/치킨_점포_상권.csv
import os,json,time,urllib.request,pandas as pd
from concurrent.futures import ThreadPoolExecutor
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/상권_분기'; os.makedirs(R,exist_ok=True)
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('SEOUL_OPENAPI_KEY_A=')][0]
def get(svc,s,q):
    for t in range(5):
        try: return json.load(urllib.request.urlopen(f'http://openapi.seoul.go.kr:8088/{K}/json/{svc}/{s}/{s+999}/{q}',timeout=90)).get(svc,{})
        except Exception: time.sleep(3*(t+1))
    raise RuntimeError(f'{svc} {q} {s} 실패')
def quarter(args):
    svc,q=args; first=get(svc,1,q); n=int(first.get('list_total_count',0)); rows=first.get('row',[])
    for s in range(1001,n+1,1000): rows+=get(svc,s,q).get('row',[])
    return [r for r in rows if r.get('SVC_INDUTY_CD')=='CS100007'],n
QS=[f'{y}{q}' for y in range(2021,2027) for q in range(1,5)]
for svc,out in [('VwsmTrdarSelngQq','치킨_매출_상권.csv'),('VwsmTrdarStorQq','치킨_점포_상권.csv')]:
    with ThreadPoolExecutor(6) as ex: res=list(ex.map(quarter,[(svc,q) for q in QS]))
    d=pd.DataFrame([r for rows,_ in res for r in rows]); d.to_csv(f'{R}/{out}',index=False,encoding='utf-8-sig')
    print(svc,'분기별 전체 행',dict(zip(QS,[n for _,n in res])),'→ 치킨 행',len(d))
