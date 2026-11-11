# 추천 단위(상권 정의) 후보의 가능성만 빠르게 확인. 깊은 모델링 아님.
#  1. 자리 단위: 같은 자리(10m)에서 개업 전 10년간 음식점이 몇 번 폐업했나 → 3년 실제 폐업
#  2. 같은 브랜드 최근접 거리(프랜차이즈) → 3년 실제 폐업
#  3. 250m 격자 스크리닝: 음식점이 있는 칸 중 '경쟁 적음 + 같은 브랜드 없음'으로 거르면 후보가 몇 칸 남나
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(cw)
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year; c['면적']=np.log1p(pd.to_numeric(c['소재지면적'],errors='coerce').fillna(40))
c['y']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
a=cw.a; al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close}).dropna(subset=['X'])
AT=cKDTree(al[['X','Y']].values); ac=al.c.values; ao=al.o.values
d=c[c.연.between(2015,2022)].copy()

# 1. 자리 이력
n=[]
for x,y,o in zip(d.X.values,d.Y.values,d.open.values):
    q=np.array(AT.query_ball_point([x,y],10),dtype=int); n.append(((ac[q]<o)&(ac[q]>=o-np.timedelta64(3650,'D'))).sum())
d['자리폐업10y']=n; d['자리구간']=pd.cut(d.자리폐업10y,[-1,0,1,2,4,999],labels=['0회','1회','2회','3–4회','5회+'])
print('[1] 같은 자리(10m) 개업 전 10년 음식점 폐업 횟수별 3년 실제 폐업률 (2015–22 개업)')
print(d.groupby(['자리구간','프랜차이즈'],observed=True).y.agg(['mean','size']).unstack().round(3).to_string())
m=smf.logit('y~프랜차이즈+면적+C(연)+np.log1p(자리폐업10y)',d).fit(disp=0); k='np.log1p(자리폐업10y)'
print(f'  로지스틱(프랜차이즈·면적·연도 통제): OR {np.exp(m.params[k]):.2f}, p={m.pvalues[k]:.4f}')

# 2. 같은 브랜드 최근접 거리
f=d[d.프랜차이즈].copy(); nd=[]
for r in f.itertuples():
    s=c[(c.브랜드==r.브랜드)&(c.open<r.open)&(c.close.isna()|(c.close>r.open))]
    nd.append(np.hypot(s.X.values-r.X,s.Y.values-r.Y).min() if len(s) else np.nan)
f['동브랜드m']=nd; f['거리구간']=pd.cut(f.동브랜드m,[0,500,1000,2000,1e6],labels=['~500m','500m–1km','1–2km','2km+'])
print('\n[2] 같은 브랜드 최근접 점포 거리별 3년 실제 폐업률 (프랜차이즈)'); print(f.groupby('거리구간',observed=True).y.agg(['mean','size']).round(3).to_string())
ff=f.dropna(subset=['동브랜드m']).copy(); ff['l거리']=np.log(ff.동브랜드m+50); ff=ff[ff.브랜드.isin(ff.브랜드.value_counts()[lambda s:s>=20].index)]
m=smf.logit('y~C(브랜드)+면적+C(연)+l거리',ff).fit(disp=0); k='l거리'; print(f'  브랜드 고정효과 로지스틱: OR {np.exp(m.params[k]):.2f}, p={m.pvalues[k]:.3f}')

# 3. 250m 격자 스크리닝 (2026-11 현재 영업 기준)
now=pd.Timestamp('2026-11-04'); live=al[al.c.isna()|(al.c>now)]; g=pd.DataFrame({'gx':(live.X//250).astype(int),'gy':(live.Y//250).astype(int)})
cell=g.value_counts().rename('음식점').reset_index(); cell=cell[cell.음식점>=10].copy(); cell['X']=cell.gx*250+125; cell['Y']=cell.gy*250+125
cl=c[c.close.isna()|(c.close>now)]; CT=cKDTree(cl[['X','Y']].values)
cell['치킨500']=[len(q) for q in CT.query_ball_point(cell[['X','Y']].values,500)]
q25=cell.치킨500.quantile(.25); print(f'\n[3] 250m 격자: 음식점 10곳 이상인 칸 {len(cell)}개 (칸 면적 6.25ha), 500m 치킨집 중앙값 {cell.치킨500.median():.0f}, 하위 25% 기준 ≤{q25:.0f}곳')
for b in ['교촌','굽네','60계','푸라닭','처갓집']:
    s=cl[cl.브랜드==b]; bd=cKDTree(s[['X','Y']].values).query(cell[['X','Y']].values)[0]
    ok=(cell.치킨500<=q25)&(bd>1000); print(f'  {b}: 현재 서울 점포 {len(s)}곳, 같은 브랜드 1km 밖 칸 {(bd>1000).sum()}개, 거기에 경쟁 하위 25%까지 걸면 {ok.sum()}개')
