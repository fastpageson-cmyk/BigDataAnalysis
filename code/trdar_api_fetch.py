# 서울시 상권분석서비스 OpenAPI 수집 (키는 .env에서 읽고 출력하지 않음) → data/raw/상권_API/
#  추정매출-상권(OA-15572, VwsmTrdarSelngQq): 2021Q1~ 분기별 상권×업종. 외식업(CS1000xx)만 저장 (치킨 비중 계산용)
#  상권변화지표-상권(OA-15576, VwsmTrdarIxQq): 2021Q1~ 분기별 상권 변화지표(LL/LH/HL/HH)·운영/폐업 영업개월 평균
#  API는 2021년 이전 분기를 돌려주지 않음(INFO-200, 2026-11-11 확인). 같은 이름의 trdar_fetch.py는 다른 세션이 만든 치킨 전용 수집본
import os,json,time,urllib.request,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
E=dict(l.strip().split('=',1) for l in open(f'{O}/.env') if '=' in l and not l.startswith('#'))
def pull(svc,key,q,keep=lambda r:True):
    rows,s=[],1
    while True:
        u=f'http://openapi.seoul.go.kr:8088/{E[key]}/json/{svc}/{s}/{s+999}/{q}'
        for t in range(4):
            try: d=json.load(urllib.request.urlopen(u,timeout=90)); break
            except Exception as e: time.sleep(3); err=e
        else: raise SystemExit(f'{svc} {q} 실패: {type(err).__name__}')
        b=d.get(svc)
        if not b: return rows,0
        rows+=[r for r in b['row'] if keep(r)]; n=b['list_total_count']
        if s+999>=n: return rows,n
        s+=1000
os.makedirs(f'{O}/data/raw/상권_API',exist_ok=True)
QS=[f'{y}{q}' for y in range(2021,2027) for q in range(1,5)]
S,I=[],[]
for q in QS:
    r,n=pull('VwsmTrdarSelngQq','SEOUL_KEY_TRDAR_SALES',q,lambda r:r['SVC_INDUTY_CD'].startswith('CS1000'))
    if not n: print(q,'매출 없음'); continue
    S+=r; r2,n2=pull('VwsmTrdarIxQq','SEOUL_KEY_TRDAR_IX',q); I+=r2
    print(q,'매출',n,'→외식',len(r),'| 변화지표',n2,flush=True)
s=pd.DataFrame(S).drop_duplicates(); i=pd.DataFrame(I).drop_duplicates()
s.to_csv(f'{O}/data/raw/상권_API/추정매출_상권_외식_2021_.csv',index=False,encoding='utf-8-sig')
i.to_csv(f'{O}/data/raw/상권_API/상권변화지표_상권_2021_.csv',index=False,encoding='utf-8-sig')
print('매출',len(s),s.STDR_YYQU_CD.min(),s.STDR_YYQU_CD.max(),'| 변화지표',len(i))
