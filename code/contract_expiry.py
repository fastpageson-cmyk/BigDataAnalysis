# 가맹 계약 만료 때 문을 더 닫나: 공정위 가맹기간(15125484)의 최초 계약연수 T를 브랜드에 붙이고,
#  서울 인허가 프랜차이즈 치킨집의 월별 실제 폐업 위험이 개업 후 T년 무렵(만료 전 3개월 ~ 후 3개월)에 높아지는지 본다.
#  비교: 같은 점포들의 만료 창 바로 앞뒤 6개월(창 전 9–4개월, 창 후 4–9개월). 다른 계약연수 브랜드의 같은 개월 수와도 비교한다.
#  실제 폐업 = 인허가 폐업 − 같은 자리 재등록(점주 교체). 관측 끝 2026-11-04.
# 출력: data/processed/contract_expiry.csv
import os,io,contextlib,importlib.util,numpy as np,pandas as pd
from scipy.stats import poisson
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
sp=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(sp)
with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(cw)
c=cw.STORES['B_넓음']; c=c[c.브랜드.notna()&c.open.dt.year.between(2010,2023)].copy()
k=pd.read_csv(f'{O}/data/raw/ftc_disclosure/contract.csv',dtype=str)
K={'교촌':'교촌','BBQ':'비비큐|BBQ','bhc':'비에이치씨','굽네':'굽네','처갓집':'처갓집','지코바':'지코바','60계':'^60계','푸라닭':'푸라닭','자담':'자담','멕시카나':'멕시카나',
   '호식이':'호식이','네네':'네네치킨','노랑통닭':'노랑통닭','깐부':'깐부','페리카나':'페리카나','또래오래':'또래오래'}
T={b:int(pd.to_numeric(k[k.brandNm.str.contains(p,na=False,regex=True)].frstCtrtYycnt,errors='coerce').mode().iloc[0]) for b,p in K.items() if k.brandNm.str.contains(p,na=False,regex=True).any()}
print('브랜드별 최초 계약연수(공정위 정보공개서):',T)
c=c[c.브랜드.isin(T)].copy(); c['T']=c.브랜드.map(T)
END=pd.Timestamp('2026-11-04'); c['end']=c.close.fillna(END).clip(upper=END)
c['ev']=c.close.notna()&(c.close<=END)&~c.재등록
c['dur']=((c.end-c.open).dt.days/30.4375).clip(lower=0)
print(f'점포 {len(c)}곳, 실제 폐업 {int(c.ev.sum())}건')
# 월별 위험: 개월 m(0..96)마다 위험 노출 점포-월, 폐업 수
M=97; ex=np.zeros((5+1,M)); ev=np.zeros((5+1,M))
for t,g in c.groupby('T'):
    d=g.dur.values; e=g.ev.values
    for m in range(M):
        ex[t,m]=np.clip(d-m,0,1).sum(); ev[t,m]=(e&(d>=m)&(d<m+1)).sum()
R=[]
for t in sorted(T and set(T.values())):
    e0=t*12; win=list(range(e0-3,e0+3)); near=list(range(e0-9,e0-3))+list(range(e0+3,e0+9))
    if max(near)>=M: continue
    a,ea=ev[t,win].sum(),ex[t,win].sum(); b,eb=ev[t,near].sum(),ex[t,near].sum()
    rr=(a/ea)/(b/eb) if b else np.nan; p=1-poisson.cdf(a-1,ea*b/eb) if b else np.nan
    # 같은 개월 수에서 계약연수가 다른 브랜드(만료가 아닌 시점)
    o=[x for x in set(T.values()) if x!=t]; oa=ev[o][:,win].sum(); oe=ex[o][:,win].sum(); ob=ev[o][:,near].sum(); oeb=ex[o][:,near].sum()
    R.append(dict(최초계약연수=t,브랜드=','.join(b for b in T if T[b]==t),만료창_월위험=a/ea*100,앞뒤_월위험=b/eb*100,배율=rr,p_한쪽=p,만료창_폐업=int(a),
                  다른브랜드_같은개월_배율=((oa/oe)/(ob/oeb) if ob else np.nan)))
R=pd.DataFrame(R); pd.set_option('display.width',220); print(R.round(3).to_string(index=False)); R.round(4).to_csv(f'{P}/contract_expiry.csv',index=False,encoding='utf-8-sig')
for t in sorted(set(T.values())):
    h=ev[t]/np.maximum(ex[t],1)*100; print(f'T={t}년 연차별 월평균 위험(%):',' '.join(f'{y+1}년차 {h[y*12:(y+1)*12].mean():.2f}' for y in range(7)))
