# OSM 도로망으로 점포 입지의 "길" 피처: 가장 가까운 길의 위계, 대로까지 거리, 300m 안 교차로 수, 근사 통행 중심성(betweenness, 표본 300개 출발점)
import os,json,numpy as np,pandas as pd,networkx as nx
from pyproj import Transformer
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W=json.load(open(f'{O}/data/raw/도로망/osm_roads_seoul.json'))['elements']
tf=Transformer.from_crs(4326,5174,always_xy=True)
RANK={'trunk':5,'primary':5,'primary_link':4,'secondary':4,'secondary_link':4,'tertiary':3,'tertiary_link':3,'unclassified':2,'residential':2,'living_street':1,'pedestrian':1,'footway':1}
G=nx.Graph(); pts=[];prk=[]
for w in W:
    g=w.get('geometry'); h=w.get('tags',{}).get('highway')
    if not g or h not in RANK: continue
    xs,ys=tf.transform([p['lon'] for p in g],[p['lat'] for p in g]); nodes=[(round(x),round(y)) for x,y in zip(xs,ys)]
    for a,b in zip(nodes[:-1],nodes[1:]):
        if a!=b: G.add_edge(a,b,length=float(np.hypot(a[0]-b[0],a[1]-b[1])))
    for x,y in zip(xs,ys): pts.append((x,y)); prk.append(RANK[h])
print('그래프 노드',G.number_of_nodes(),'엣지',G.number_of_edges())
# 차수 2 노드는 그대로 두고 근사 betweenness 계산
bc=nx.betweenness_centrality(G,k=300,weight='length',seed=0,normalized=True)
N=np.array(list(bc.keys())); Bv=np.array(list(bc.values())); TN=cKDTree(N)
deg=np.array([G.degree(tuple(n)) for n in N]); inter=N[deg>=3]; TI=cKDTree(inter)
P=np.array(pts); PR=np.array(prk); TP=cKDTree(P); big=P[PR>=4]; TB=cKDTree(big)
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl'); XY=s[['X','Y']].values
d,i=TP.query(XY); s_rank=PR[i]
# 30m 안 가장 높은 위계(대로변인가)
near=TP.query_ball_point(XY,30); top=np.array([PR[q].max() if len(q) else 0 for q in near])
F=pd.DataFrame({'길_접한최고위계':top,'길_대로까지m':np.log1p(TB.query(XY)[0]),'길_교차로300m':[len(q) for q in TI.query_ball_point(XY,300)],
                '길_통행중심성':np.log1p(1e5*np.array([Bv[q].max() if len(q) else 0 for q in TN.query_ball_point(XY,50)]))},index=s.index)
F.to_pickle(f'{O}/data/processed/street_feats.pkl'); print(F.describe().T.round(2).to_string())
