# 지하철 출입구(OSM, ref=출구번호) → 가장 가까운 역에 연결, 인허가 좌표계(EPSG:5174)로 변환
import os,json,numpy as np,pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
e=pd.DataFrame([{'osm_id':x['id'],'lat':x['lat'],'lon':x['lon'],'출구':x.get('tags',{}).get('ref')} for x in json.load(open(f'{O}/data/raw/출입구/osm_subway_entrance_seoul.json'))['elements']])
st=pd.read_csv(f'{O}/data/raw/출입구/seoul_station_master_20260902.csv',encoding='cp949'); st.columns=['역ID','역명','호선','위도','경도']
st=st.groupby('역명',as_index=False)[['위도','경도']].mean()   # 환승역은 한 점으로
tf=Transformer.from_crs(4326,5174,always_xy=True)
st['X'],st['Y']=tf.transform(st.경도.values,st.위도.values); e['X'],e['Y']=tf.transform(e.lon.values,e.lat.values)
d,i=cKDTree(st[['X','Y']].values).query(e[['X','Y']].values); e['역명']=st.역명.values[i]; e['역거리m']=d.round()
e=e[e.역거리m<=400].copy(); e['출구']=e.출구.astype(str).str.extract(r'^(\d+)')[0]
e=e.dropna(subset=['출구']).drop_duplicates(['역명','출구'])
e.to_csv(f'{O}/data/processed/exits.csv',index=False,encoding='utf-8-sig')
c=e.groupby('역명').size(); print('출구',len(e),'역',len(c),'역당 중앙',c.median())
for n in ['강남','홍대입구','여의도','잠실','성수','노원','신설동','역삼','신촌','건대입구']: print(n, sorted(e[e.역명==n].출구.astype(int).tolist()))
