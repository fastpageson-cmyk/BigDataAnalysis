# 웹 지도(web/index.html)용 데이터 → web/data.json, data/processed/grid_candidates_v2.csv
#  구 경계(행정동 경계를 구로 합침), 상권 폴리곤 + 매출×폐업 사분면(quadrant.py), 250m 칸(음식점 10곳 이상, 현재 2026-11-04 기준)
#  후보 규칙(웹과 같음): 같은 브랜드 1km 밖 + 상권 점포당 매출 상위 25%. '가늘고 길게'는 여기에 가게 회전 상위 25% 아님 + 500m 치킨집 중앙값 이하
#  칸 변수: 500m 치킨집 수, 칸 중심이 속한 상권의 점포당 매출(최근 2024–26H1)·사분면, 가게 회전(최근 5년 2년 못 간 음식점 폐업 / 현재 음식점), 브랜드별 같은 브랜드 최근접 거리
#  브랜드 밀집: 브랜드별 현재 영업 점포를 500m 칸으로 센 값(2곳 이상 칸만). 점포 좌표는 내보내지 않음
import os,io,json,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from pyproj import Transformer
from shapely.geometry import shape,mapping
from shapely.ops import unary_union,transform as stf
from shapely import points
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'; WEB=f'{O}/web'; os.makedirs(WEB,exist_ok=True)
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign'); A=aa.A
END=pd.Timestamp('2026-11-04'); FAR=pd.Timestamp('2100-01-01'); S=250
c=cw.STORES['B_넓음'].copy(); c['cc']=c.close.fillna(FAR)
a=cw.a; al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close.fillna(FAR)}).dropna(subset=['X','o'])
t4=Transformer.from_crs(5174,4326,always_xy=True)
def r5(g): return json.loads(json.dumps(mapping(g),separators=(',',':')),parse_float=lambda x:round(float(x),5))

# 1. 구 경계
tf=Transformer.from_crs(5179,4326,always_xy=True).transform
g=json.load(open(f'{O}/data/raw/행정동경계/HangJeongDong_ver20200701.geojson')); F=[f for f in g['features'] if f['properties']['sido']=='11']
gu={}
for f in F: gu.setdefault(f['properties']['sggnm'],[]).append(shape(f['geometry']))
GU=[{'name':k,'geom':r5(stf(tf,unary_union(v)).simplify(0.0003))} for k,v in gu.items()]

# 2. 상권 + 사분면(최근 기간 기준) — quadrant.py 결과
W=pd.read_csv(f'{P}/quadrant_trdar.csv',dtype={'상권코드':str}).set_index('상권코드')
mx,my=W.점포당매출_뒤.median(),W.폐업률_뒤.median()
W['q']=np.select([(W.점포당매출_뒤>=mx)&(W.폐업률_뒤<my),(W.점포당매출_뒤>=mx),(W.폐업률_뒤<my)],[1,2,3],4)
W['both1']=(W.사분면.str.startswith('①')&(W.q==1)).astype(int)
W['매출분위']=W.점포당매출_뒤.rank(pct=True)
tA=Transformer.from_crs(5181,4326,always_xy=True).transform
TR=[]
for i,geom in enumerate(aa.G):
    code=str(A.상권코드.iloc[i]); w=W.loc[code] if code in W.index else None
    TR.append({'code':code,'name':A.상권명.iloc[i],'type':A.상권유형.iloc[i],'gu':A.구명.iloc[i],'geom':r5(stf(tA,geom).simplify(0.00008)),
               **({'q':int(w.q),'both1':int(w.both1),'sales':round(float(w.점포당매출_뒤)/1e4,2),'sales0':round(float(w.점포당매출_앞)/1e4,2),
                   'close':round(float(w.폐업률_뒤)*100,1),'close0':round(float(w.폐업률_앞)*100,1),'pct':round(float(w.매출분위),3)} if w is not None else {})})

# 3. 칸
live=al[(al.o<=END)&(al.c>END)]; gx,gy=(live.X//S).astype(int),(live.Y//S).astype(int)
cell=pd.DataFrame({'gx':gx,'gy':gy}).value_counts().rename('음식점').reset_index(); cell=cell[cell.음식점>=10].copy()
cell['X']=cell.gx*S+S/2; cell['Y']=cell.gy*S+S/2
alive=c[(c.open<=END)&(c.cc>END)]; CT=cKDTree(c[['X','Y']].values); am=((c.open<=END)&(c.cc>END)).values
cell['치킨500']=[am[q].sum() for q in CT.query_ball_point(cell[['X','Y']].values,500)]
dead=al[(al.c>=END-pd.DateOffset(years=5))&(al.c<=END)&((al.c-al.o).dt.days<730)]
dc=pd.DataFrame({'gx':(dead.X//S).astype(int),'gy':(dead.Y//S).astype(int)}).value_counts().rename('단명폐업')
cell=cell.join(dc,on=['gx','gy']).fillna({'단명폐업':0}); cell['회전율']=cell.단명폐업/cell.음식점
gi,_,_=aa.assign(cell.X.values,cell.Y.values); cell['상권코드']=np.where(gi>=0,A.상권코드.astype(str).values[np.clip(gi,0,None)],None)
cell=cell.join(W[['점포당매출_뒤','q','매출분위']],on='상권코드')
cell['lon'],cell['lat']=t4.transform(cell.X.values,cell.Y.values)
bc=alive.브랜드.value_counts(); BR=[b for b in bc.index if bc[b]>=20]
ft=pd.read_csv(f'{P}/chicken_brand_table.csv').set_index('브랜드'); qb=pd.read_csv(f'{P}/quadrant_brand.csv').set_index('브랜드')
dist={}; dens={}
for b in BR:
    s=alive[alive.브랜드==b]; dist[b]=np.minimum(cKDTree(s[['X','Y']].values).query(cell[['X','Y']].values)[0],9999).round(-1).astype(int).tolist()
    k=pd.DataFrame({'x':(s.X//500).astype(int),'y':(s.Y//500).astype(int)}).value_counts(); k=k[k>=2]
    lon,lat=t4.transform((k.index.get_level_values(0)*500+250).values,(k.index.get_level_values(1)*500+250).values)
    dens[b]=[[round(a_,5),round(b_,5),int(n)] for a_,b_,n in zip(lat,lon,k.values)]
q25,q50=cell.치킨500.quantile(.25),cell.치킨500.quantile(.5); turn75=cell.회전율.quantile(.75)
brands=[{'name':b,'stores':int(bc[b]),'close3':(round(float(ft.loc[b,'서울3년실제폐업'])*100,1) if b in ft.index else None),
         'sales':(round(float(ft.loc[b,'평균매출24_천원'])/1e5,2) if b in ft.index and pd.notna(ft.loc[b,'평균매출24_천원']) else None),
         'cost':(round(float(ft.loc[b,'창업비_천원'])/1e5,2) if b in ft.index and pd.notna(ft.loc[b,'창업비_천원']) else None),
         'quad':(qb.loc[b,'사분면_뒤'] if b in qb.index else None)} for b in BR]
CELLS={'lat':cell.lat.round(5).tolist(),'lon':cell.lon.round(5).tolist(),'food':cell.음식점.astype(int).tolist(),'chk':cell.치킨500.astype(int).tolist(),
       'turn':cell.회전율.round(3).tolist(),'tr':cell.상권코드.fillna('').tolist(),'sales':[None if pd.isna(x) else round(x/1e4,2) for x in cell.점포당매출_뒤],
       'pct':[None if pd.isna(x) else round(x,3) for x in cell.매출분위],'q':[None if pd.isna(x) else int(x) for x in cell.q]}
meta={'asof':'2026-11-04','q25':float(q25),'q50':float(q50),'turn75':round(float(turn75),3),'cells':len(cell)}
json.dump({'meta':meta,'gu':GU,'trdar':TR,'cells':CELLS,'brands':brands,'dist':dist,'dens':dens},open(f'{WEB}/data.json','w'),ensure_ascii=False,separators=(',',':'))
print(f'칸 {len(cell)}개, 브랜드 {len(BR)}개: {BR}\n경쟁 하위25% {q25:.0f}곳, 중앙값 {q50:.0f}곳, 회전율 상위25% 기준 {turn75:.3f}, data.json {os.path.getsize(f"{WEB}/data.json")/1e6:.1f}MB')

# 4. 격자 후보 v2 (웹과 같은 규칙) → CSV
rows=[]
for b in BR:
    d=np.array(dist[b]); base=(d>1000)&cell.q.notna().values
    # grid_v2_backtest.py 결과: 상권 매출 상위 25%만 검증됨(2024–26 폐업 14.2% vs 21.4%, p=0.003). 경쟁 하위 25%는 프랜차이즈에서 폐업↑라 뺌
    stable=base&(cell.매출분위.values>=.75)&(cell.회전율.values<turn75)&(cell.치킨500.values<=q50)   # 가늘고 길게
    bold=base&(cell.매출분위.values>=.75)                                                          # 굵고 짧게
    for nm,m in [('가늘고 길게',stable),('굵고 짧게',bold)]:
        t=cell[m].assign(브랜드=b,전략=nm,동브랜드최근접m=d[m]); rows.append(t)
    print(f'  {b}: 가늘고 길게 {stable.sum()}칸, 굵고 짧게 {bold.sum()}칸')
v=pd.concat(rows); v=v.join(A.set_index(A.상권코드.astype(str))[['상권명','구명']],on='상권코드')
v['칸ID']=v.gx.astype(str)+'_'+v.gy.astype(str)
v[['브랜드','전략','칸ID','구명','상권명','lat','lon','음식점','치킨500','회전율','점포당매출_뒤','동브랜드최근접m']].round(4).to_csv(f'{P}/grid_candidates_v2.csv',index=False,encoding='utf-8-sig')
