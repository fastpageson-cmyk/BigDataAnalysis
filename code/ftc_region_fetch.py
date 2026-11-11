# 공정위 가맹정보 '브랜드 가맹점 및 직영점정보'(data.go.kr 15125490): 브랜드×지역(시도)×회계연도 가맹점·직영점 수. 업종 필터가 없어 전부 받고 치킨만 저장.
#  연도별로 data/raw/ftc_region/치킨_{yr}.csv 저장 → 이미 있는 연도는 건너뜀(이어받기). 키는 .env에서 읽고 출력하지 않음.
import os,json,time,urllib.parse,urllib.request,urllib.error,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{O}/data/raw/ftc_region'; os.makedirs(D,exist_ok=True)
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('DATA_GO_KR_KEY2=')][0]
U='https://apis.data.go.kr/1130000/FftcBrandFrcsDropInfo3_Service/getbrandFrcsDmsstus2'
for yr in range(2017,2026):
    f=f'{D}/치킨_{yr}.csv'
    if os.path.exists(f): continue
    rows=[]; p=1; tot=None
    while True:
        q=urllib.parse.urlencode(dict(serviceKey=K,pageNo=p,numOfRows=1000,resultType='json',jngBizCrtraYr=yr))
        for t in range(5):
            try: d=json.load(urllib.request.urlopen(U+'?'+q,timeout=90)); break
            except urllib.error.HTTPError as e:
                if e.code==429: raise SystemExit(f'{yr} p{p}: 하루 한도(429). 내일 다시 실행하면 이 연도부터 이어받음')
                time.sleep(5); err=e
            except Exception as e: time.sleep(5); err=e
        else: raise SystemExit(f'{yr} p{p} 실패: {type(err).__name__}')
        it=d.get('items') or []
        if isinstance(it,dict): it=it.get('item',[])
        rows+=[x for x in it if x.get('indutyMlsfcNm')=='치킨']; tot=int(d.get('totalCount') or 0)
        if p*1000>=tot or not it: break
        p+=1
    pd.DataFrame(rows).to_csv(f,index=False,encoding='utf-8-sig'); print(yr,'전체',tot,'치킨',len(rows),flush=True)
