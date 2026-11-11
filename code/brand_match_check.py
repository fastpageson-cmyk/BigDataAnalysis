# 브랜드 매칭 검수(과제 요구 '수동 검수'의 자동 보조): 공정위가 집계한 서울 가맹점+직영점 수(15125490, ftc_region_fetch.py)와
#  인허가 상호 정규식으로 센 서울 영업 점포 수(chicken_wide.py B_넓음, 회계연도 말 기준)를 브랜드×연도로 비교한다.
#  비율 = 인허가 매칭 수 / 공정위 서울 수. 1보다 작으면 상호에 브랜드명이 없는 점포(간판 전환·영업자 승계)를 놓친 것, 크면 오매칭 의심.
# 출력: data/processed/brand_match_check.csv
import os,io,glob,contextlib,importlib.util,numpy as np,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
sp=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(sp)
with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(cw)
c=cw.STORES['B_넓음']; FAR=pd.Timestamp('2100-01-01')
R=pd.concat([pd.read_csv(f,dtype=str) for f in sorted(glob.glob(f'{O}/data/raw/ftc_region/치킨_*.csv'))])
R=R[R.areaNm=='서울'].copy(); R['acntgYr']=R.acntgYr.astype(int); R['n']=pd.to_numeric(R.allFrcsDmsCnt,errors='coerce')
R=R[R.n>0]   # 0은 그해 서울 행이 없거나 미기재(BBQ 2016, 네네 2020 등) → 비교에서 뺌
L=pd.read_csv(f'{O}/data/raw/ftc_brand_list_2017_2025.csv',dtype=str)[['brandMnno','brandNm']].drop_duplicates('brandMnno')
R=R.merge(L,on='brandMnno',how='left').drop_duplicates(['brandMnno','acntgYr'])
K={'교촌':'교촌','BBQ':'비비큐|BBQ','bhc':'비에이치씨|BHC','굽네':'굽네','처갓집':'처갓집','지코바':'지코바','60계':'^60계','푸라닭':'푸라닭','자담':'자담','멕시카나':'멕시카나',
   '호식이':'호식이','네네':'네네치킨','노랑통닭':'노랑통닭','깐부':'깐부','페리카나':'페리카나','또래오래':'또래오래'}
rows=[]
for b,pat in K.items():
    r=R[R.brandNm.fillna('').str.contains(pat,regex=True)]
    for yr,g in r.groupby('acntgYr'):
        t=pd.Timestamp(f'{yr}-12-31'); s=c[(c.브랜드==b)&(c.open<=t)&(c.close.fillna(FAR)>t)]
        rows.append(dict(브랜드=b,회계연도=yr,공정위_서울=int(g.n.sum()),인허가_매칭=len(s)))
d=pd.DataFrame(rows); d['비율']=(d.인허가_매칭/d.공정위_서울).round(3)
pd.set_option('display.width',200); print(d.pivot(index='브랜드',columns='회계연도',values='비율').round(2).to_string())
s=d.groupby('브랜드')[['공정위_서울','인허가_매칭']].sum(); s['비율']=(s.인허가_매칭/s.공정위_서울).round(3); print('\n전 기간 합'); print(s.sort_values('비율').to_string())
print(f"\n전체: 인허가 매칭 {d.인허가_매칭.sum()} / 공정위 서울 {d.공정위_서울.sum()} = {d.인허가_매칭.sum()/d.공정위_서울.sum():.3f}")
d.to_csv(f'{P}/brand_match_check.csv',index=False,encoding='utf-8-sig')
