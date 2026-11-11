# 점포→행정동 매핑(상가정보 지번 기준, adv.py 방식) + 동네 변수
#  ① 인허가로 만든 시점별 동 변수(개업 직전 연말 기준, 2004~): 음식점 수·3년 증가율·1년 폐업률·개업률·카페 비중
#  ② 상권분석서비스(2021Q1~): 외식 매출/점포, 유동·직장·상주인구, 집객시설 → 개업 직전 분기 값(없으면 2021Q1 값, 플래그)
import os,glob,numpy as np,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/'; RAW=R+'maximus-analysis-archive/data/raw/'
sb=pd.read_csv(R+'데이터/원천/서울캠/상가정보/sbiz_store_info_seoul_202606.csv',usecols=['시군구명','행정동코드','법정동명','지번본번지','지번부번지'],low_memory=False)
sb['gd']=sb['시군구명']+' '+sb['법정동명']; sb['bo']=sb['지번본번지'].astype('Int64').astype(str); sb['bu']=sb['지번부번지'].fillna(0).astype('Int64').astype(str)
maj=lambda k: sb.groupby(k+['행정동코드']).size().reset_index(name='c').sort_values('c',ascending=False).drop_duplicates(k)[k+['행정동코드']]
c3,c2,c1=maj(['gd','bo','bu']),maj(['gd','bo']),maj(['gd'])
def adm(df):
    a=df['지번주소'].fillna('').str.extract(r'서울특별시\s+(\S+구)\s+(\S+?(?:동\d*가|동|가|로\d*가|로))\s+(?:산\s*)?(\d+)(?:-(\d+))?')
    t=pd.DataFrame({'gd':a[0]+' '+a[1],'bo':a[2],'bu':a[3].fillna('0')},index=df.index).reset_index()
    m=t.merge(c3.rename(columns={'행정동코드':'a3'}),how='left',on=['gd','bo','bu']).merge(c2.rename(columns={'행정동코드':'a2'}),how='left',on=['gd','bo']).merge(c1.rename(columns={'행정동코드':'a1'}),how='left',on='gd').set_index('index')
    return m.a3.fillna(m.a2).fillna(m.a1)
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl')
a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['adm']=adm(a); print('전체 음식점 행정동 매핑률',a.adm.notna().mean().round(3))
a=a[a.adm.notna()]; a['adm']=a.adm.astype('int64'); a['cafe']=a['업태구분명'].isin(['커피숍','까페','떡카페','전통찻집'])
op=a.open; cl=a.close.fillna(pd.Timestamp('2100-01-01'))
rec=[]
for Y in range(2004,2026):
    e=pd.Timestamp(f'{Y}-12-31'); s=e-pd.DateOffset(years=1)
    x=a[(op<=e)&(cl>e)]; g=x.groupby('adm')
    r=pd.DataFrame({'동_음식점수':g.size(),'동_카페수':g.cafe.sum()})
    r['동_개업1y']=a[(op>s)&(op<=e)].groupby('adm').size(); r['동_폐업1y']=a[(cl>s)&(cl<=e)].groupby('adm').size(); r['Y']=Y
    rec.append(r.fillna(0).reset_index())
p=pd.concat(rec).set_index(['adm','Y']).sort_index()
p['동_3y증가']=p.동_음식점수/p.동_음식점수.groupby('adm').shift(3)-1
p['동_폐업률']=p.동_폐업1y/p.동_음식점수; p['동_개업률']=p.동_개업1y/p.동_음식점수; p['동_카페비중']=p.동_카페수/p.동_음식점수
p=p[['동_음식점수','동_3y증가','동_폐업률','동_개업률','동_카페비중']].reset_index()
p.to_pickle(f'{O}/data/processed/_dong_lic_yearly.pkl')
s=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); s['adm']=adm(s); print('브랜드 점포 매핑률',s.adm.notna().mean().round(3))
s.to_pickle(f'{O}/data/processed/_brand_stores_y.pkl')
# ② 상권분석서비스
def rd(f):
    for e in ['utf-8','cp949','utf-8-sig']:
        try: return pd.read_csv(f,encoding=e,low_memory=False)
        except Exception: pass
sl=pd.concat([rd(f) for f in sorted(glob.glob(f'{O}/data/raw/상권_행정동/sales_*.csv'))+sorted(glob.glob(RAW+'seoul_commercial_sales_dong_202*.csv'))])
st=pd.concat([rd(f) for f in sorted(glob.glob(f'{O}/data/raw/상권_행정동/stores_*.csv'))+sorted(glob.glob(RAW+'seoul_commercial_stores_dong_202*.csv'))])
FOOD_S=sl.columns[sl.columns.str.contains('서비스_업종_코드_명')][0]; FOOD_T=st.columns[st.columns.str.contains('서비스_업종_코드_명')][0]
print('업종 예시',sl[FOOD_S].unique()[:12])
sl=sl[sl.행정동_코드.notna()]; key=['기준_년분기_코드','행정동_코드']
food=sl[sl.columns[sl.columns.str.contains('서비스_업종_코드$')][0]].astype(str).str.startswith('CS1')
m=sl[food].groupby(key)['당월_매출_금액'].sum().rename('동_외식매출')
stc=[c for c in st.columns if c in('점포_수','전체_점포_수')]
st['_n']=st[stc].bfill(axis=1).iloc[:,0]
fs=st[st[st.columns[st.columns.str.contains('서비스_업종_코드$')][0]].astype(str).str.startswith('CS1')].groupby(key)['_n'].sum().rename('동_외식점포')
k=pd.concat([m,fs],axis=1); k['동_점포당매출']=k.동_외식매출/k.동_외식점포
for f,c,nm in [('seoul_commercial_floating_population_dong.csv','총_유동인구_수','동_유동인구'),('seoul_workplace_population_dong.csv','총_직장_인구_수','동_직장인구'),('seoul_resident_population_dong.csv','총_상주인구_수','동_상주인구'),('seoul_attracting_facilities_dong.csv','집객시설_수','동_집객시설')]:
    d=rd(RAW+f); k=k.join(d.groupby(key)[c].sum().rename(nm),how='outer')
k=k.reset_index().rename(columns={'행정동_코드':'adm','기준_년분기_코드':'yq'}); k['adm']=k.adm.astype('int64')
k.to_pickle(f'{O}/data/processed/_dong_market_q.pkl'); print(k.groupby('yq').size().to_dict()); print(k.describe().T[['count','mean']].round(1))
