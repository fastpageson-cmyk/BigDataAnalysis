# 점포 단위 정답지 시험: 공정위 가맹점 목록(15125492, 2022 회계연도, 서울분)의 좌표를 인허가 음식점 좌표에 붙여
#  ① 인허가 상호 정규식이 실제 가맹점을 얼마나 잡는지(재현율) ② 상호에 브랜드명이 없는 가맹점이 얼마나 되는지 본다.
#  목록은 브랜드 32개, 서울 283곳뿐(정보공개서에 목록을 낸 브랜드만)이라 '일부 표본 검수'로만 쓴다.
#  붙이는 규칙: 2022-12-31에 영업 중인 인허가 음식점 중 50m 안 가장 가까운 곳. 좌표는 EPSG:5174(인허가)로 변환.
import os,io,re,contextlib,importlib.util,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
sp=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(sp)
with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(cw)
a=cw.a.copy(); a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce')
t=pd.Timestamp('2022-12-31'); live=a[(a.open<=t)&(a.close.fillna(pd.Timestamp('2100-01-01'))>t)&a.X.notna()].reset_index(drop=True)
s=pd.read_csv(f'{O}/data/raw/ftc_disclosure/stores.csv',dtype=str); s=s[s.areaNm=='서울'].copy()
s['lon']=pd.to_numeric(s.wdthCrdntVal,errors='coerce'); s['lat']=pd.to_numeric(s.vrticlCrdntVal,errors='coerce')
print(f'공정위 목록 서울 {len(s)}곳(브랜드 {s.brandNm.nunique()}), 좌표 있음 {s.lon.notna().sum()}')
s=s[s.lon.between(126.7,127.3)&s.lat.between(37.4,37.75)].copy()
X,Y=Transformer.from_crs(4326,5174,always_xy=True).transform(s.lon.values,s.lat.values)
d,i=cKDTree(live[['X','Y']].values).query(np.c_[X,Y])
s['거리m']=d.round(1); s['인허가상호']=live.nm.values[i]; s['인허가브랜드']=live.브랜드.values[i]
s['붙음']=s.거리m<=50
# 가장 가까운 한 곳은 같은 건물의 다른 가게일 수 있어, 50m 안에 같은 브랜드로 잡힌 인허가가 하나라도 있는지도 본다(주요 브랜드만)
BM={'BBQ':'BBQ','교촌치킨':'교촌'}; T=cKDTree(live[['X','Y']].values)
s['50m안_같은브랜드']=[(live.브랜드.values[q]==BM.get(b)).any() if b in BM else np.nan for b,q in zip(s.brandNm,T.query_ball_point(np.c_[X,Y],50))]
def key(b): return re.sub(r'\(.*?\)|치킨|통닭|양념|두마리','',b).replace(' ','')[:3]
s['상호에브랜드']=[(key(b) in str(n)) or (b.upper()[:3] in str(n).upper()) for b,n in zip(s.brandNm,s.인허가상호)]
m=s[s.붙음]
print(f'50m 안에 영업 중 음식점이 붙은 곳 {len(m)}/{len(s)} ({len(m)/len(s):.0%}), 거리 중앙값 {s.거리m.median():.0f}m')
print(f'붙은 곳 중 인허가 상호에 브랜드명이 있는 곳 {m.상호에브랜드.mean():.1%}, 우리 정규식이 치킨 브랜드로 잡은 곳 {m.인허가브랜드.notna().mean():.1%}')
for b in ['BBQ','교촌치킨']:
    x=s[s.brandNm==b]; print(f'{b}: 공정위 서울 목록 {len(x)}곳 중 50m 안에 우리 정규식의 같은 브랜드 인허가가 있는 곳 {x["50m안_같은브랜드"].mean():.1%}')
g=m.groupby('brandNm').agg(n=('붙음','size'),상호에브랜드=('상호에브랜드','mean')).query('n>=5').sort_values('n',ascending=False); print(g.round(2).to_string())
print('\n상호에 브랜드명이 없는 예(상호는 인허가 공개값):'); print(m[~m.상호에브랜드][['brandNm','인허가상호','거리m']].head(12).to_string(index=False))
s[['brandNm','거리m','붙음','상호에브랜드','인허가브랜드','50m안_같은브랜드']].to_csv(f'{P}/brand_store_match.csv',index=False,encoding='utf-8-sig')
