# 데모: 대표 상권 중심점에 "지금(2026-11) 새로 낸다면" 브랜드별 5년 내 실제 폐점 위험 예측 → 업종별 추천 순위
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.spatial import cKDTree
from pyproj import Transformer
from sklearn.linear_model import LogisticRegression
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); H=5
import sys,io,contextlib
with contextlib.redirect_stdout(io.StringIO()): from rec_eval import s,PROF,COMP,DL,ST
TN={0:'시장·장년생활권',1:'오피스',2:'가족·주거여가',3:'2030핫플',4:'야간번화가'}
# 학습: 2008~2021-11 개업(5년 관측 완료)
d=s[(s.open.dt.year>=2008)&(s.open<=END-pd.DateOffset(years=H))&s.상권코드.notna()&s['외식매출(log)'].notna()].copy()
y=((d.close<=d.open+pd.DateOffset(years=H))&d.진짜폐점).astype(int)
br=y.groupby(d.브랜드).agg(['sum','count']); pr=y.mean(); enc=(br['sum']+20*pr)/(br['count']+20); d['브랜드률']=d.브랜드.map(enc)
CAT=sorted(d.업종.unique()); TOPB=d.브랜드.value_counts().index[:15]
def design(D):
    X=pd.concat([D[['브랜드률']+ST+COMP+DL+PROF],pd.get_dummies(D.업종,dtype=float).reindex(columns=CAT,fill_value=0)],axis=1); return X
X=design(d); MED=X.median(); X=X.fillna(MED); LO=X.quantile(.01); HI=X.quantile(.99); MU=X.mean(); SD=X.std().replace(0,1)
def zi(X,D):
    Z=((X.fillna(MED).clip(LO,HI,axis=1))-MU)/SD   # 학습 범위(1~99%) 밖 값은 잘라 외삽 방지
    for b in TOPB:
        bb=(D.브랜드==b).astype(float).values
        for c in PROF: Z[f'{b}×{c}']=bb*Z[c].values
    return Z
m=LogisticRegression(C=0.05,max_iter=5000).fit(zi(X,d),y)
# 현재 시점 경쟁 변수 계산용 인덱스
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl'); a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce'); a=a[a.X.notna()&a.open.notna()].reset_index(drop=True)
T=cKDTree(a[['X','Y']].values); ao=a.open.values; ac=a.close.fillna(pd.Timestamp('2100-01-01')).values; acafe=(a.src=='cafe').values
B=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); B=B[(B.n_match==1)&B.X.notna()].reset_index(drop=True)
TB=cKDTree(B[['X','Y']].values); bo=B.open.values; bc=B.close.fillna(pd.Timestamp('2100-01-01')).values; bb=B.브랜드.values; bg=B.업종.values
def comp(x,yy,brand,cat,t=np.datetime64(END)):
    t1=t-np.timedelta64(365,'D'); p=np.array(T.query_ball_point([x,yy],300),dtype=int); live=(ao[p]<t)&(ac[p]>t); p5=np.array(T.query_ball_point([x,yy],500),dtype=int)
    q=np.array(TB.query_ball_point([x,yy],500),dtype=int); lb=(bo[q]<t)&(bc[q]>t); q1=np.array(TB.query_ball_point([x,yy],1000),dtype=int); lb1=(bo[q1]<t)&(bc[q1]>t); sb=lb1&(bb[q1]==brand)
    return dict(밀도300=live.sum(),휴게밀도300=(live&acafe[p]).sum(),폐업1y500=((ac[p5]<=t)&(ac[p5]>t1)).sum(),신규1y500=((ao[p5]<=t)&(ao[p5]>t1)).sum(),
        같은브랜드500=(lb&(bb[q]==brand)).sum(),같은브랜드1km=sb.sum(),같은업종타브랜드500=(lb&(bg[q]==cat)&(bb[q]!=brand)).sum(),같은업종신규1y500=((bo[q]<=t)&(bo[q]>t1)&(bg[q]==cat)&(bb[q]!=brand)).sum(),
        최근접같은브랜드m=np.min(np.hypot(B.X.values[q1[sb]]-x,B.Y.values[q1[sb]]-yy)) if sb.any() else 1500)
A=pd.read_pickle(f'{O}/data/processed/areas.pkl'); P=pd.read_pickle(f'{O}/data/processed/area_profile.pkl'); dl=pd.read_pickle(f'{O}/data/processed/_dong_lic_yearly.pkl')
tf=Transformer.from_crs(5181,5174,always_xy=True)
brands=d.groupby(['업종','브랜드']).size(); brands=brands[brands>=60].reset_index()[['업종','브랜드']]
AREAS=['강남역','여의도역(여의도)','잠실역','노원역','신설동역','홍대입구역(홍대)','성수역']
out=[]
for nm in AREAS:
    r=A[A.상권명==nm].iloc[0]; x,yy=tf.transform(r.XCNTS_VALU,r.YDNTS_VALU); pf=P.loc[r.상권코드]
    sub=s[s.상권코드==r.상권코드]; adm=sub.adm.mode().iloc[0] if len(sub) else np.nan
    dv=dl[(dl.adm==adm)&(dl.Y==2025)].iloc[0] if len(sub) else None
    rows=[]
    for c,b in brands.itertuples(index=False):
        f=comp(x,yy,b,c); f.update({k:pf[k] for k in PROF}); f.update({k:dv[k] for k in DL}); f['동_음식점수']=np.log1p(f['동_음식점수']); f.update(dict(면적=np.log1p(55),휴게=int(c in('커피','베이커리·디저트')),다중=0,브랜드=b,업종=c,브랜드률=enc[b]))
        rows.append(f)
    D=pd.DataFrame(rows); D['위험5y']=m.predict_proba(zi(design(D),D))[:,1]; D['기본위험']=D.브랜드률; D['상권']=nm; D['유형']=TN[int(pf.유형번호)]
    out.append(D)
o=pd.concat(out); o.to_csv(f'{O}/data/processed/rec_demo.csv',index=False,encoding='utf-8-sig')
for c in ['커피','치킨']:
    t=o[o.업종==c].pivot_table(index='브랜드',columns='상권',values='위험5y').round(3); t['브랜드평균']=o[o.업종==c].groupby('브랜드').기본위험.first().round(3)
    print(f'\n## {c}: 5년 내 실제 폐점 예측 위험 (낮을수록 추천)'); print(t.sort_values('브랜드평균').to_string())
    rk=o[o.업종==c].assign(순위=lambda z:z.groupby('상권').위험5y.rank()).pivot_table(index='브랜드',columns='상권',values='순위'); print(rk.astype(int).to_string())
print(o.groupby('상권').유형.first().to_dict())
