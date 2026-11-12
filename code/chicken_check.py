# 이원중 제안(50대 치킨 창업: 동네 경쟁 밀도·프랜차이즈 여부 → 3년 폐업) 빠른 검증
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl'); a=a[a.src=='rest'].copy()
a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['nm']=a['사업장명'].fillna('').str.replace(r'\s+','',regex=True)
src=open(f'{O}/code/wide.py',encoding='utf-8').read(); B=eval(src[src.index('B={')+2:src.index('\nR=[]')])['치킨']
a['브랜드']=None
for b,p in B.items(): a.loc[a.nm.str.contains(p,regex=True)&a.브랜드.isna(),'브랜드']=b
ck=a[(a.업태구분명=='통닭(치킨)')|((a.업태구분명=='호프/통닭')&a.nm.str.contains('치킨|통닭|닭강정'))|a.브랜드.notna()].copy()
ck=ck[~ck.nm.str.contains('코리안BBQ|바베큐|비비큐보쌈',regex=True)|ck.브랜드.notna()]
ck['X']=pd.to_numeric(ck['좌표정보(X)'],errors='coerce'); ck['Y']=pd.to_numeric(ck['좌표정보(Y)'],errors='coerce')
ck['프랜차이즈']=ck.브랜드.notna()
# 같은 자리(30m) 1년 안 치킨집 재등록 = 점주 교체
c=ck[ck.X.notna()].reset_index(drop=True); T=cKDTree(c[['X','Y']].values); same=np.zeros(len(c),bool)
for i,r in c[c.close.notna()].iterrows():
    q=[j for j in T.query_ball_point([r.X,r.Y],30) if j!=i]
    if q: o=c.open.values[q]; same[i]=((o>=np.datetime64(r.close-pd.Timedelta(days=60)))&(o<=np.datetime64(r.close+pd.Timedelta(days=90)))).any()
c['재등록']=same
d=c[c.open.dt.year.between(2015,2022)].copy()
d['y_신고']=(d.close<=d.open+pd.DateOffset(years=3)).astype(int); d['y_실제']=(d.y_신고.astype(bool)&~d.재등록).astype(int)
print(f'2015–22 개업 치킨집 {len(d)}곳 (프랜차이즈 {d.프랜차이즈.sum()}), 3년 폐업 신고 {d.y_신고.mean():.3f}, 그중 같은 자리 재등록 제외 {d.y_실제.mean():.3f}')
print(d.groupby('프랜차이즈')[['y_신고','y_실제']].mean().round(3).assign(n=d.groupby('프랜차이즈').size()).to_string())
# 경쟁 밀도: 개업 시점 500m 안 영업 중 치킨집 수 (행정동 대신 반경, 동 단위도 함께)
ao=c.open.values; ac=c.close.fillna(pd.Timestamp('2100-01-01')).values; dens=[]
for r in d.itertuples():
    q=np.array(T.query_ball_point([r.X,r.Y],500),dtype=int); t=np.datetime64(r.open); dens.append(((ao[q]<t)&(ac[q]>t)).sum())
d['치킨밀도500']=dens; d['연']=d.open.dt.year; d['면적']=np.log1p(pd.to_numeric(d['소재지면적'],errors='coerce').fillna(40))
d['밀도구간']=pd.qcut(d.치킨밀도500,5,labels=['최저','낮음','중간','높음','최고'],duplicates='drop')
print(d.groupby('밀도구간',observed=True)[['y_신고','y_실제']].mean().round(3).assign(n=d.groupby('밀도구간',observed=True).size(),평균밀도=d.groupby('밀도구간',observed=True).치킨밀도500.mean().round(1)).to_string())
for y in ['y_신고','y_실제']:
    m=smf.logit(f'{y}~np.log1p(치킨밀도500)+프랜차이즈+면적+C(연)',d).fit(disp=0)
    print(f'{y}: 밀도(log) OR {np.exp(m.params["np.log1p(치킨밀도500)"]):.2f} p={m.pvalues["np.log1p(치킨밀도500)"]:.3f} | 프랜차이즈 OR {np.exp(m.params["프랜차이즈[T.True]"]):.2f} p={m.pvalues["프랜차이즈[T.True]"]:.3f} | 면적(log) OR {np.exp(m.params["면적"]):.2f} p={m.pvalues["면적"]:.3f}')
# 기준선 대비 예측력(시간 분리 2015–19 학습 / 2020–22 평가)
tr=d.연<=2019; te=~tr
for y in ['y_신고','y_실제']:
    m=smf.logit(f'{y}~np.log1p(치킨밀도500)+프랜차이즈+면적',d[tr]).fit(disp=0); p=m.predict(d[te])
    pf=d[tr].groupby('프랜차이즈')[y].mean(); pb=d[te].프랜차이즈.map(pf)
    print(f'{y} 평가 AUC: 프랜차이즈 여부만 {roc_auc_score(d[te][y],pb):.3f} | +밀도+면적 {roc_auc_score(d[te][y],p):.3f} | 다수클래스 정확도 {1-d[te][y].mean():.3f}')
