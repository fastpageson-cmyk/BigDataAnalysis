# 출구 단위 분석 가능성: ① 표본 ② 출구 거리 효과(브랜드·연도 고정) ③ 같은 역 안 출구 간 폐점률 차이가 우연 이상인가(순열검정)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); H=5
ex=pd.read_csv(f'{O}/data/processed/exits.csv'); s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl')
d,i=cKDTree(ex[['X','Y']].values).query(s[['X','Y']].values); s['출구거리m']=d; s['역']=ex.역명.values[i]; s['출구']=ex.역명.values[i]+' '+ex.출구.astype(str).values[i]+'번'
s.to_pickle(f'{O}/data/processed/store_feats.pkl')
e=s[(s.open.dt.year>=2008)&(s.open<=END-pd.DateOffset(years=H))].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=H))&e.진짜폐점).astype(int)
print(f'5년 관측 점포 {len(e)}, 실제폐점 {e.y.sum()}')
for r in [100,150,200,300]: print(f'출구 {r}m 안: {(e.출구거리m<=r).mean():.1%}')
w=e[e.출구거리m<=200]; c=w.groupby('출구').size(); cb=w.groupby(['출구','브랜드']).size()
print(f'200m 안 점포 {len(w)}, 출구 {c.size}개, 출구당 점포 중앙 {c.median():.0f}, 10곳 이상 출구 {(c>=10).sum()}개 | 출구×브랜드 칸 {cb.size}개 중 3곳 이상 {(cb>=3).sum()}개')
print('강남역 출구별 점포(5년 관측분):',w[w.역=='강남'].groupby('출구').size().to_dict())
# ② 출구 거리 효과
e['거리구간']=pd.cut(e.출구거리m,[0,100,200,300,500,1e9],labels=['~100m','100-200','200-300','300-500','500m+'])
e['기대']=e.groupby('브랜드').y.transform('mean'); t=e.groupby('거리구간',observed=True).agg(n=('y','size'),폐점률=('y','mean'),브랜드기대=('기대','mean')); t['브랜드대비']=t.폐점률/t.브랜드기대; print(t.round(3).to_string())
e['logd']=np.log(e.출구거리m.clip(lower=20)); m=smf.logit('y~logd+C(브랜드)+C(연)',e).fit(disp=0,method='bfgs',maxiter=500)
print(f'출구거리 2배 → 폐점 오즈 {np.exp(m.params.logd*np.log(2)):.2f}배, p={m.pvalues.logd:.3f}')
# ③ 같은 역 안 출구 간 이질성: 브랜드 기대 대비 관측 폐점의 출구별 편차 제곱합을, 역 안에서 출구 라벨을 섞은 분포와 비교
w=e[e.출구거리m<=200].copy(); w=w[w.groupby('역').출구.transform('nunique')>=2]
def stat(df): g=df.groupby('출구').agg(o=('y','sum'),x=('기대','sum')); return (((g.o-g.x)**2)/g.x.clip(lower=0.05)).sum()
obs=stat(w); rs=np.random.RandomState(0); sim=[]
for _ in range(1000):
    v=w.copy(); v['출구']=v.groupby('역').출구.transform(lambda z: rs.permutation(z.values)); sim.append(stat(v))
print(f'역 내 출구 이질성: 관측 {obs:.1f}, 섞은 분포 중앙 {np.median(sim):.1f}, 95% {np.percentile(sim,95):.1f}, p={(np.sum(np.array(sim)>=obs)+1)/1001:.3f} (n={len(w)}, 역 {w.역.nunique()}, 출구 {w.출구.nunique()})')
