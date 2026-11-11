# 점포 → 서울시 상권분석서비스 상권(영역) 배정. 인허가 좌표 EPSG:5174, 상권 경계 EPSG:5181
import os,numpy as np,pandas as pd,shapefile
from shapely.geometry import shape,Point
from shapely import STRtree,points
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
r=shapefile.Reader(f'{O}/data/raw/상권영역/area',encoding='utf-8')
A=pd.DataFrame(r.records(),columns=[f[0] for f in r.fields[1:]]).rename(columns={'TRDAR_SE_1':'상권유형','TRDAR_CD':'상권코드','TRDAR_CD_N':'상권명','SIGNGU_CD_':'구명','ADSTRD_CD':'adm8','ADSTRD_CD_':'행정동명','RELM_AR':'면적m2'})
G=[shape(s.__geo_interface__) for s in r.shapes()]; tree=STRtree(G)
tf=Transformer.from_crs(5174,5181,always_xy=True)
def assign(x,y):
    gx,gy=tf.transform(x,y); P=points(gx,gy); out=np.full(len(x),-1)
    ok=~np.isnan(gx); idx=tree.query(P[ok],predicate='within')   # (점,폴리곤) 쌍
    o=np.where(ok)[0]; t=pd.DataFrame({'p':o[idx[0]],'g':idx[1],'ar':A.면적m2.values[idx[1]]}).sort_values('ar').drop_duplicates('p')  # 겹치면 작은 상권
    out[t.p.values]=t.g.values; return out,gx,gy
if __name__=='__main__':
    s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl')
    g,gx,gy=assign(s.X.values,s.Y.values); s['gi']=g; s['gx']=gx; s['gy']=gy
    s=s.join(A[['상권유형','상권코드','상권명']],on='gi')
    print('상권 배정률',(g>=0).mean().round(3)); print(s.상권유형.value_counts(dropna=False).to_dict())
    s.to_pickle(f'{O}/data/processed/store_feats.pkl'); A.to_pickle(f'{O}/data/processed/areas.pkl')
    print(s[s.상권명.isin(['강남역','성수역','홍대입구역(홍대)'])].groupby(['상권명','업종']).size().unstack(fill_value=0))
