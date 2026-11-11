# 격자 후보 규칙 v2(web_map_data.py와 같은 규칙) 검증: 2023-12-31 상태로 칸을 판정 → 2024-01~2026-11 실제 폐업 비교
#  안정형(가늘고 길게): 500m 치킨집 하위 25% + 가게 회전 상위 25% 아님 + 상권 점포당 매출(2021–23) 상위 50% + 같은 브랜드 1km 밖
#  매출형(굵고 짧게): 상권 점포당 매출 상위 25% + 500m 치킨집 중앙값 이하 + 같은 브랜드 1km 밖
#  대상: 2024-01-01에 영업 중이고 음식점 10곳 이상 칸·매출 자료 있는 상권에 있는 치킨집. y = 기간 안 실제 폐업(재등록 제외)
#  비교: 단순 폐업률, 로지스틱(브랜드·연차·면적 통제), 규칙 요소별 효과
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign'); A=aa.A
T=pd.Timestamp('2023-12-31'); END=pd.Timestamp('2026-11-04'); FAR=pd.Timestamp('2100-01-01'); S=250
c=cw.STORES['B_넓음'].copy().reset_index(drop=True); c['cc']=c.close.fillna(FAR)
a=cw.a; al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close.fillna(FAR)}).dropna(subset=['X','o'])
live=al[(al.o<=T)&(al.c>T)]; cell=pd.DataFrame({'gx':(live.X//S).astype(int),'gy':(live.Y//S).astype(int)}).value_counts().rename('음식점')
dead=al[(al.c>=T-pd.DateOffset(years=5))&(al.c<=T)&((al.c-al.o).dt.days<730)]
dc=pd.DataFrame({'gx':(dead.X//S).astype(int),'gy':(dead.Y//S).astype(int)}).value_counts().rename('단명')
cell=pd.concat([cell,dc],axis=1).fillna(0); cell=cell[cell.음식점>=10].copy(); cell['회전율']=cell.단명/cell.음식점
cell['X']=cell.index.get_level_values(0)*S+S/2; cell['Y']=cell.index.get_level_values(1)*S+S/2
am=((c.open<=T)&(c.cc>T)).values; CT=cKDTree(c[['X','Y']].values)
cell['치킨500']=[am[q].sum() for q in CT.query_ball_point(cell[['X','Y']].values,500)]
q25,q50,t75=cell.치킨500.quantile(.25),cell.치킨500.quantile(.5),cell.회전율.quantile(.75)
W=pd.read_csv(f'{P}/quadrant_trdar.csv',dtype={'상권코드':str}).set_index('상권코드'); W['pct']=W.점포당매출_앞.rank(pct=True)
gi,_,_=aa.assign(cell.X.values,cell.Y.values); cell['상권코드']=np.where(gi>=0,A.상권코드.astype(str).values[np.clip(gi,0,None)],None); cell=cell.join(W.pct,on='상권코드')
s=c[am].copy(); s['gx'],s['gy']=(s.X//S).astype(int),(s.Y//S).astype(int)
s=s.join(cell[['치킨500','회전율','pct']],on=['gx','gy']).dropna(subset=['치킨500','pct'])
br=s.브랜드.values; d=np.full(len(s),9999.)
for b in pd.unique(br[pd.notna(br)]):
    o=c[am&(c.브랜드==b).values]; k=np.where(br==b)[0]
    if len(o)>1: dd,_=cKDTree(o[['X','Y']].values).query(s.iloc[k][['X','Y']].values,2); d[k]=dd[:,1]   # 자기 자신 제외
s['동브랜드m']=d; s['y']=((s.cc<=END)&~s.재등록).astype(int); s['연차']=np.log1p((T-s.open).dt.days/365.25)
s['면적']=np.log1p(pd.to_numeric(s.소재지면적,errors='coerce').fillna(40)); s['프랜']=s.프랜차이즈.astype(int)
s['안정형']=((s.치킨500<=q25)&(s.회전율<t75)&(s.pct>=.5)&(s.동브랜드m>1000)).astype(int)
s['매출형']=((s.pct>=.75)&(s.치킨500<=q50)&(s.동브랜드m>1000)).astype(int)
s['경쟁낮음']=(s.치킨500<=q25).astype(int); s['회전낮음']=(s.회전율<t75).astype(int); s['매출상위50']=(s.pct>=.5).astype(int); s['매출상위25']=(s.pct>=.75).astype(int); s['브랜드1km밖']=(s.동브랜드m>1000).astype(int)
print(f'2024-01-01 영업 중 치킨집 {len(s)}곳(프랜차이즈 {s.프랜.sum()}), 2024-01~2026-11 실제 폐업률 {s.y.mean():.3f}. 기준: 경쟁 하위25% {q25:.0f}곳, 중앙값 {q50:.0f}곳, 회전 상위25% {t75:.3f}')
f=s[s.프랜==1]; f=f[f.브랜드.map(f.브랜드.value_counts())>=10]
for nm,g,fx in [('전체',s,'프랜+연차+면적'),('프랜차이즈',f,'C(브랜드)+연차+면적'),('개인',s[s.프랜==0],'연차+면적')]:
    out=[]
    for r in ['안정형','매출형','경쟁낮음','회전낮음','매출상위50','매출상위25']+(['브랜드1km밖'] if nm=='프랜차이즈' else []):
        m=smf.logit(f'y~{fx}+{r}',g).fit(disp=0,method='bfgs',maxiter=2000)
        out.append(f'{r}: 통과 {g[g[r]==1].y.mean():.3f}({int(g[r].sum())}) vs {g[g[r]==0].y.mean():.3f}, OR {np.exp(m.params[r]):.2f} p={m.pvalues[r]:.3f}')
    print(f'\n[{nm}] n={len(g)}, 폐업률 {g.y.mean():.3f}'); print('  '+'\n  '.join(out))

# 수정안(위 결과를 본 뒤 정한 규칙이라 사후 규칙으로 표시): 경쟁 하위 25% 조건은 프랜차이즈에서 폐업↑(OR 2.63)이라 뺌
s['안정형2']=((s.pct>=.5)&(s.회전율<t75)&(s.치킨500<=q50)).astype(int)   # 매출 상위 50% + 회전 낮음 + 경쟁 중앙값 이하
s['매출형2']=((s.pct>=.75)&(s.회전율<t75)).astype(int)                    # 매출 상위 25% + 회전 낮음
print('\n[수정안, 사후 규칙]')
for nm,g,fx in [('전체',s,'프랜+연차+면적'),('프랜차이즈',f.join(s[['안정형2','매출형2']]),'C(브랜드)+연차+면적'),('개인',s[s.프랜==0],'연차+면적')]:
    for r in ['안정형2','매출형2']:
        m=smf.logit(f'y~{fx}+{r}',g).fit(disp=0,method='bfgs',maxiter=2000)
        print(f'  {nm} {r}: 통과 {g[g[r]==1].y.mean():.3f}({int(g[r].sum())}) vs {g[g[r]==0].y.mean():.3f}, OR {np.exp(m.params[r]):.2f} p={m.pvalues[r]:.3f}')
