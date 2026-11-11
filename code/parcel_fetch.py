# 치킨집(chicken_wide.py B_넓음, 2010–2023 개업) 지번 필지에 붙일 공개 자료 수집. 키는 .env에서 읽고 출력하지 않음
#  1) 서울시 개별공시지가(OA-1180, IndividuallyPostedLandPriceService): 필지 × (개업 전년) 공시지가 원/㎡ → data/raw/필지/jiga.csv
#  2) 국토부 건축HUB 건축물대장 표제부(data.go.kr 15134735, getBrTitleInfo): 필지의 건물 동별 층수·연면적·사용승인일·주용도 → data/raw/필지/brtitle.csv
#  이미 받은 필지는 건너뜀(이어받기). 소유자 등 개인정보 항목은 저장하지 않음
import os,io,re,json,time,contextlib,importlib.util,urllib.request,urllib.error,urllib.parse,pandas as pd
from concurrent.futures import ThreadPoolExecutor
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/필지'; os.makedirs(R,exist_ok=True)
ENV=dict(l.strip().split('=',1) for l in open(f'{O}/.env') if '=' in l and not l.startswith('#'))
KS,KD=ENV['SEOUL_OPENAPI_KEY_C'],ENV['DATA_GO_KR_KEY2']
ADDR=r'서울특별시\s+(\S+구)\s+(\S+?(?:동\d*가|동|가|로\d*가|로))\s+(산\s*)?(\d+)(?:-(\d+))?'
def parcels():
    sp=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(cw)
    c=cw.STORES['B_넓음']; c=c[c.open.dt.year.between(2010,2023)].copy()
    m=c.지번주소.fillna('').str.extract(ADDR); c['구'],c['동'],c['산'],c['본'],c['부']=m[0],m[1],m[2].notna().astype(int),m[3],m[4].fillna('0')
    c['기준연']=c.open.dt.year-1
    return c.dropna(subset=['구'])
class Quota(Exception): pass
def get(u,tries=4):
    for t in range(tries):
        try: return json.load(urllib.request.urlopen(u,timeout=60))
        except urllib.error.HTTPError as e:
            if e.code==429: raise Quota()   # 하루 호출 한도 초과: 저장하고 멈춤
            time.sleep(2*(t+1))
        except Exception: time.sleep(2*(t+1))
    return None
def jiga(row):
    g,d,s,b,j,y=row; q=lambda *p:'/'.join(urllib.parse.quote(str(x)) for x in p)
    r=get(f'http://openapi.seoul.go.kr:8088/{KS}/json/IndividuallyPostedLandPriceService/1/5/'+q(g,d,int(b),int(j),2 if s else 1,y))
    rows=(r or {}).get('IndividuallyPostedLandPriceService',{}).get('row',[])
    x=rows[0] if rows else {}
    return dict(구=g,동=d,산=s,본=b,부=j,기준연=y,공시지가=x.get('JIGA'),시군구코드=x.get('SIGUNGU_CD'),법정동코드=x.get('BJDONG_CD'),ok=r is not None)
KEEP=['sigunguCd','bjdongCd','bun','ji','platGbCd','mgmBldrgstPk','regstrKindCdNm','mainAtchGbCdNm','mainPurpsCdNm','etcPurps','grndFlrCnt','ugrndFlrCnt','totArea','archArea','platArea','useAprDay','strctCdNm','hhldCnt','fmlyCnt','rideUseElvtCnt']
def title(key):
    sg,bj,s,b,j=key; out=[]; page=1
    while True:
        r=get('https://apis.data.go.kr/1613000/BldRgstHubService/getBrTitleInfo?'+urllib.parse.urlencode({'serviceKey':KD,'sigunguCd':sg,'bjdongCd':bj,'platGbCd':'1' if str(s)=='1' else '0','bun':f'{int(b):04d}','ji':f'{int(j):04d}','numOfRows':'100','pageNo':str(page),'_type':'json'}))
        if r is None: return [dict(sigunguCd=sg,bjdongCd=bj,bun=b,ji=j,platGbCd=s,조회실패=1)]
        body=r.get('response',{}).get('body',{}); it=(body.get('items') or {}).get('item',[]) if isinstance(body.get('items'),dict) else []
        it=[it] if isinstance(it,dict) else it; out+= [{k:x.get(k) for k in KEEP} for x in it]
        if page*100>=int(body.get('totalCount',0) or 0): break
        page+=1
    return out or [dict(sigunguCd=sg,bjdongCd=bj,bun=b,ji=j,platGbCd=s,건물없음=1)]
if __name__=='__main__':
    c=parcels(); K=['구','동','산','본','부','기준연']; todo=c[K].drop_duplicates()
    fj=f'{R}/jiga.csv'; done=pd.read_csv(fj,dtype=str) if os.path.exists(fj) else pd.DataFrame(columns=K)
    seen=set(map(tuple,done[K].astype(str).values)); todo=[tuple(x) for x in todo.astype(str).values if tuple(x) not in seen]
    todo=[(g,d,int(s),b,j,int(y)) for g,d,s,b,j,y in todo]; print(f'공시지가: 필지×연 {len(todo)}건 새로 조회')
    with ThreadPoolExecutor(8) as ex: res=list(ex.map(jiga,todo))
    J=pd.concat([done,pd.DataFrame(res).astype(str)]); J.to_csv(fj,index=False,encoding='utf-8-sig')
    J=J[J.ok=='True']; print(f'  공시지가 있음 {(J.공시지가!="None").mean():.3f}')
    keys=J[J.법정동코드!='None'][['시군구코드','법정동코드','산','본','부']].drop_duplicates()
    fb=f'{R}/brtitle.csv'; bd=pd.read_csv(fb,dtype=str) if os.path.exists(fb) else pd.DataFrame(columns=['sigunguCd','bjdongCd','platGbCd','bun','ji'])
    seen=set(zip(bd.sigunguCd,bd.bjdongCd,bd.platGbCd.astype(str),bd.bun.astype(str).str.lstrip('0'),bd.ji.astype(str).str.lstrip('0')))
    kt=[tuple(x) for x in keys.values if (x[0],x[1],str(x[2]),str(int(x[3])) if int(x[3]) else '',str(int(x[4])) if int(x[4]) else '') not in seen]
    print(f'건축물대장: 필지 {len(kt)}곳 새로 조회 (200곳마다 저장, 하루 한도 초과 시 멈춤 → 다음 날 다시 실행하면 이어받음)')
    buf=[]
    def flush():
        global bd
        if buf: bd=pd.concat([bd,pd.DataFrame(buf).astype(str)]); bd.to_csv(fb,index=False,encoding='utf-8-sig'); buf.clear()
    try:
        for n,k in enumerate(kt,1):
            buf+=title(k); time.sleep(0.05)
            if n%200==0: flush(); print(f'  {n}/{len(kt)}',flush=True)
    except Quota: print('  하루 호출 한도 초과(429). 받은 데까지 저장하고 멈춤')
    flush(); print(f'  건물 행 {len(bd)}, 조회 끝난 필지 {bd.drop_duplicates(["sigunguCd","bjdongCd","platGbCd","bun","ji"]).shape[0]}')
