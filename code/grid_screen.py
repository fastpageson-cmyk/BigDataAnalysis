# 250m 격자 선별: 모든 브랜드가 같은 격자(EPSG:5174 기준 250m 칸)를 쓰고, 브랜드마다 달라지는 것은 '어느 칸이 규칙을 통과하나'뿐이다.
# 규칙 (모두 칸 중심 기준, 판단 시점에 영업 중인 점포로 계산)
#   R1 경쟁: 칸 중심 500m 안 치킨집 수가 그 시점 '음식점 10곳 이상 칸'의 하위 Q 분위 이하
#   R2 자기잠식: 칸 중심 D m 안에 같은 브랜드 점포 없음
# 1) 백테스트: 2018–22 개업 프랜차이즈 치킨집을 개업 시점 기준으로 통과/탈락 칸에 나눠 3년 실제 폐업률 비교(브랜드 고정효과)
#    기준값을 바꿔(Q 25·50%, D 500·1000·1500m) 결과가 유지되는지 확인
# 2) 현재(2026-11-04) 브랜드별 후보 칸 목록 → data/processed/grid_candidates.csv (칸 단위, 점포 정보 없음)
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
spec=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(cw)
S=250; END=pd.Timestamp('2026-11-04'); FAR=pd.Timestamp('2100-01-01')
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year; c['면적']=np.log1p(pd.to_numeric(c['소재지면적'],errors='coerce').fillna(40))
c['y']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int); c['cc']=c.close.fillna(FAR)
a=cw.a; al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close.fillna(FAR)}).dropna(subset=['X','o'])
cell_of=lambda X,Y:((X//S).astype(int),(Y//S).astype(int))
CT=cKDTree(c[['X','Y']].values)

def snapshot(t):
    """시점 t의 '음식점 10곳 이상 칸'과 칸 중심 500m 영업 중 치킨집 수"""
    live=al[(al.o<=t)&(al.c>t)]; gx,gy=cell_of(live.X,live.Y)
    cell=pd.DataFrame({'gx':gx,'gy':gy}).value_counts().rename('음식점').reset_index(); cell=cell[cell.음식점>=10].copy()
    cell['X']=cell.gx*S+S/2; cell['Y']=cell.gy*S+S/2
    alive=((c.open<=t)&(c.cc>t)).values
    cell['치킨500']=[alive[q].sum() for q in CT.query_ball_point(cell[['X','Y']].values,500)]
    return cell.set_index(['gx','gy'])

def brand_dist(t,brand,X,Y):
    s=c[(c.브랜드==brand)&(c.open<t)&(c.cc>t)]
    return cKDTree(s[['X','Y']].values).query(np.c_[X,Y])[0] if len(s) else np.full(len(X),np.inf)

# ---------- 1. 백테스트 ----------
f=c[c.프랜차이즈&c.연.between(2018,2022)].copy(); f=f[f.브랜드.isin(f.브랜드.value_counts()[lambda s:s>=20].index)]
f['gx'],f['gy']=cell_of(f.X,f.Y)
snaps={y:snapshot(pd.Timestamp(f'{y-1}-12-31')) for y in range(2018,2023)}   # 개업 전년 말 상태
rows=[]
for r in f.itertuples():
    sn=snaps[r.연]; k=(r.gx,r.gy)
    if k not in sn.index: rows.append((np.nan,np.nan,np.nan)); continue
    cx,cy=k[0]*S+S/2,k[1]*S+S/2
    rows.append((sn.at[k,'치킨500'],(sn.치킨500.rank(pct=True)).at[k],brand_dist(r.open,r.브랜드,[cx],[cy])[0]))
f[['칸치킨500','칸경쟁분위','칸동브랜드m']]=rows
fb=f.dropna(subset=['칸치킨500']).copy()
print(f'[1] 백테스트: 2018–22 개업 프랜차이즈 {len(f)}곳 중 음식점 10곳 이상 칸에 있는 {len(fb)}곳, 3년 실제 폐업률 {fb.y.mean():.3f}')
res=[]
for Q in (.25,.5):
    for D in (500,1000,1500):
        fb['R1']=fb.칸경쟁분위<=Q; fb['R2']=fb.칸동브랜드m>D; fb['통과']=fb.R1&fb.R2
        m=smf.logit('y~C(브랜드)+면적+C(연)+통과',fb).fit(disp=0); k='통과[T.True]'
        res.append(dict(경쟁분위=f'하위{int(Q*100)}%',자기잠식=f'{D}m',통과칸점포=int(fb.통과.sum()),통과_폐업률=fb[fb.통과].y.mean(),
                        탈락_폐업률=fb[~fb.통과].y.mean(),OR=np.exp(m.params[k]),p=m.pvalues[k]))
print(pd.DataFrame(res).round(3).to_string(index=False))
for nm,col,D in [('R1 경쟁만(하위25%)','R1',None),('R2 자기잠식만(1km)','R2',1000)]:
    fb['R1']=fb.칸경쟁분위<=.25; fb['R2']=fb.칸동브랜드m>1000
    m=smf.logit(f'y~C(브랜드)+면적+C(연)+{col}',fb).fit(disp=0); k=f'{col}[T.True]'
    print(f'  {nm}: 통과 {fb[fb[col]].y.mean():.3f} vs 탈락 {fb[~fb[col]].y.mean():.3f}, OR {np.exp(m.params[k]):.2f}, p={m.pvalues[k]:.3f}')

# ---------- 2. 현재 후보 칸 ----------
now=snapshot(END); q25=now.치킨500.quantile(.25); to4326=Transformer.from_crs(5174,4326,always_xy=True)
lon,lat=to4326.transform(now.X.values,now.Y.values); now['경도']=lon.round(5); now['위도']=lat.round(5)
out=[]
for b in ['교촌','지코바','굽네','처갓집','60계','푸라닭','bhc','BBQ']:
    bd=brand_dist(END,b,now.X.values,now.Y.values); ok=(now.치킨500<=q25)&(bd>1000)
    t=now[ok].assign(브랜드=b,동브랜드최근접m=bd[ok.values].round(0)); out.append(t)
    print(f'  {b}: 통과 칸 {ok.sum()}개')
g=pd.concat(out).reset_index(); g['칸ID']=g.gx.astype(str)+'_'+g.gy.astype(str)
g[['브랜드','칸ID','위도','경도','음식점','치킨500','동브랜드최근접m']].to_csv(f'{P}/grid_candidates.csv',index=False,encoding='utf-8-sig')
print(f'[2] 현재 기준 경쟁 하위 25% = 500m 치킨집 {q25:.0f}곳 이하, 후보 칸 저장: data/processed/grid_candidates.csv ({len(g)}행)')
