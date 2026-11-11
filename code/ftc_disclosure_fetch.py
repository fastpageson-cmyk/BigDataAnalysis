# 공정위 정보공개서 계열 API 수집(data.go.kr, 키 DATA_GO_KR_KEY2, 11/12 활용신청 승인).
#  ① 가맹본부 재무(15125456): 연도별 전체(브랜드 파라미터 불필요) → data/raw/ftc_disclosure/finance.csv
#  ② 가맹기간(15125484)·③ 입지선정기준(15125488): 치킨 브랜드×연도별 호출 → contract.csv, siting.csv
#  ④ 가맹점 목록(15125492): 치킨 브랜드×연도별. 시험 결과 2023 등록분(2022 회계연도)만 값이 있고 주소는 시군구까지 → stores.csv
#  대표자명(frcsRprsvNm 등)은 저장하지 않음. 이미 받은 (서비스, 연도, 브랜드)는 건너뜀.
import os,json,time,urllib.parse,urllib.request,urllib.error,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); D=f'{O}/data/raw/ftc_disclosure'; os.makedirs(D,exist_ok=True)
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('DATA_GO_KR_KEY2=')][0]
B='https://apis.data.go.kr/1130000/'
def get(path,**p):
    out=[]; pg=1
    while True:
        q=urllib.parse.urlencode(dict(serviceKey=K,pageNo=pg,numOfRows=1000,resultType='json',**p))
        for t in range(5):
            try: d=json.loads(urllib.request.urlopen(B+path+'?'+q,timeout=90).read()); break
            except urllib.error.HTTPError as e:
                if e.code==429: raise SystemExit('하루 한도(429). 내일 다시 실행하면 이어받음')
                time.sleep(4); err=e
            except Exception as e: time.sleep(4); err=e
        else: raise SystemExit(f'{path} {p} 실패: {type(err).__name__}')
        it=d.get('items') or []; it=it.get('item',[]) if isinstance(it,dict) else it
        out+=[{k:v for k,v in x.items() if 'Rprsv' not in k} for x in it]
        if pg*1000>=int(d.get('totalCount') or 0) or not it: return out
        pg+=1
def save(name,rows,keys):
    f=f'{D}/{name}.csv'; old=pd.read_csv(f,dtype=str) if os.path.exists(f) else pd.DataFrame()
    new=pd.concat([old,pd.DataFrame(rows).astype(str)]).drop_duplicates(); new.to_csv(f,index=False,encoding='utf-8-sig'); return new
# ① 재무
f=f'{D}/finance.csv'; done=set(pd.read_csv(f,dtype=str).jngBizCrtraYr) if os.path.exists(f) else set()
for yr in range(2017,2026):
    if str(yr) in done: continue
    r=get('FftcjnghdqrtrsFnnrInfo3_Service/getjnghdqrtrsFnlttinfo2',jngBizCrtraYr=yr); save('finance',r,None); print('재무',yr,len(r),flush=True)
# ②③④ 치킨 브랜드
L=pd.read_csv(f'{O}/data/raw/ftc_brand_list_2017_2025.csv',dtype=str); L=L[L.indutyMlsfcNm=='치킨']
# 호출 수를 줄이려고 가맹점 30곳 이상인 적이 있는 치킨 브랜드만(공정위 가맹점 현황 기준). 전체 4,913 브랜드-연도 → 1,437
F=pd.read_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv',dtype={'yr':str}); big=set(F[(F.indutyMlsfcNm=='치킨')&(F.frcsCnt>=30)].brandNm)
L=L[L.brandNm.isin(big)][['jngBizCrtraYr','brandMnno']].drop_duplicates()
S={'stores':('FftcbrandfrcslistinfoService/getbrandFrcsListinfo',[2023]),
   'contract':('FftcBrandFrcsStrtInfo2_Service/getbrandFrcsPdinfo',range(2017,2026)),
   'siting':('FftcbrandlcsltcrtrainfoService/getbrandLcsltCrtrainfo',range(2024,2026))}
for name,(path,yrs) in S.items():
    f=f'{D}/{name}_done.txt'; done=set(open(f).read().split()) if os.path.exists(f) else set()
    todo=L[L.jngBizCrtraYr.astype(int).isin(list(yrs))]; buf=[]; n=0
    for yr,bm in todo.values:
        k=f'{yr}_{bm}'
        if k in done: continue
        buf+=get(path,jngBizCrtraYr=yr,brandMnno=bm); done.add(k); n+=1
        if n%100==0: save(name,buf,None); buf=[]; open(f,'w').write('\n'.join(done)); print(name,n,'/',len(todo),flush=True)
    if buf: save(name,buf,None)
    open(f,'w').write('\n'.join(done)); print(name,'끝',n,flush=True)
