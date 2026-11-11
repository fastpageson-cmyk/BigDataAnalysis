# 공정위 외식업종 통합현황(data.go.kr 15165094) 연도별 전체 수집. 키는 .env에서 읽고 출력하지 않음
import os,json,time,urllib.parse,urllib.request,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K=[l.split('=',1)[1].strip() for l in open(f'{O}/.env') if l.startswith('DATA_GO_KR_KEY2=')][0]
U='https://apis.data.go.kr/1130000/FftcBrandFrcsBzmnIntgInfoEatoutTpbizStats2_Service/getBrandFrcsBzmnIntgInfoEatoutTpbiz2'
rows=[]
for yr in range(2017,2026):
    p=1
    while True:
        q=urllib.parse.urlencode(dict(serviceKey=K,pageNo=p,numOfRows=1000,resultType='json',jngBizCrtrYr=yr))
        try: d=json.load(urllib.request.urlopen(U+'?'+q,timeout=60))
        except Exception as e: print(yr,p,'오류',type(e).__name__); break
        it=d.get('items') or []
        rows+=it; n=int(d.get('totalCount') or 0)
        if p*1000>=n or not it: break
        p+=1; time.sleep(0.2)
    print(yr,'총',n)
df=pd.DataFrame(rows); df.to_csv(f'{O}/data/raw/ftc_eatout_2017_2025.csv',index=False,encoding='utf-8-sig'); print(df.shape)
