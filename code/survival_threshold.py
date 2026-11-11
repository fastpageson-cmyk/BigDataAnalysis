# '마의 구간': 치킨집이 몇 년(어떤 조건)을 넘기면 그 뒤로 안정적인가 (chicken_wide.py B_넓음, 실제 폐업 = 같은 자리 재등록 제외)
#  [1] 영업 연차별 연 폐업 위험(생명표): 2010년 이후 관측된 점포-년. 개인/프랜차이즈, 개업 시기 반반(2005–14 / 2015–23)
#  [2] 조건부 생존: a년을 버틴 점포가 다음 3년 안에 닫을 확률. 위험이 '바닥'에 닿는 연차 = 이후 연차 평균의 1.2배 안으로 처음 들어오는 해
#  [3] 구간 나누기: 연차별 위험에 1~3개 꺾임점 계단 모형을 맞춰(가중 제곱오차) 꺾이는 연차를 데이터로 찾음
#  [4] 조건: 상권 매출(quadrant.py 상권 점포당 매출 4분위, 2021–23) × 연차, 브랜드 상·하위 × 연차
import os,io,contextlib,importlib.util,itertools,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
bf=load('brand_fit'); c=bf.c.copy(); A=bf.aa.A
END=pd.Timestamp('2026-11-04'); W0=pd.Timestamp('2010-01-01'); MAXA=15
c['상권코드']=np.where(bf.gi>=0,A.상권코드.astype(str).values[np.clip(bf.gi,0,None)],None)
c['end']=c.close.fillna(END).clip(upper=END); c['dead']=c.close.notna()&(c.close<=END)&~c.재등록
c=c[(c.end>W0)&(c.open<END)].copy()
def life(g):
    """연차 a(0=첫해)별 노출 점포-년과 폐업 수 (2010-01 이후만 관측: 왼쪽 절단 처리)"""
    o=g.open.values.astype('datetime64[D]'); e=g.end.values.astype('datetime64[D]'); dd=g.dead.values; rows=[]
    for a in range(MAXA+1):
        s=o+np.timedelta64(int(365.25*a),'D'); t=o+np.timedelta64(int(365.25*(a+1)),'D')
        lo=np.maximum(s,np.datetime64(W0.date())); hi=np.minimum(t,e); ex=np.clip((hi-lo).astype(float)/365.25,0,None)
        k=(dd&(e>=lo)&(e<t)&(e>=s)).sum(); rows.append((a,ex.sum(),k))
    L=pd.DataFrame(rows,columns=['연차','점포년','폐업']).set_index('연차'); L['위험']=L.폐업/L.점포년
    L['하한'],L['상한']=[L.위험-1.96*np.sqrt(L.폐업)/L.점포년,L.위험+1.96*np.sqrt(L.폐업)/L.점포년]; return L
G={'전체':c,'개인':c[~c.프랜차이즈],'프랜차이즈':c[c.프랜차이즈],'2005–14 개업':c[c.open.dt.year.between(2005,2014)],'2015–23 개업':c[c.open.dt.year.between(2015,2023)]}
T={k:life(g) for k,g in G.items()}
print('[1] 영업 연차별 연 실제 폐업 위험 (%, 괄호 점포-년)')
print(pd.DataFrame({k:(v.위험*100).round(1).astype(str)+'('+v.점포년.round(0).astype(int).astype(str)+')' for k,v in T.items()}).to_string())

print('\n[2] a년을 버틴 점포가 다음 3년 안에 닫을 확률 (연차별 위험으로 계산)')
def cond3(L,a): h=L.위험.reindex(range(a,a+3)).values; return 1-np.prod(1-h)
print(pd.DataFrame({k:[round(cond3(v,a)*100,1) for a in range(0,11)] for k,v in T.items()},index=[f'{a}년 버팀' for a in range(0,11)]).to_string())
for k,v in T.items():
    h=v.위험; fl=[a for a in range(0,MAXA-2) if h.loc[a]<=1.2*h.loc[a:].mean() and all(h.loc[b]<=1.2*h.loc[a:].mean() for b in range(a,min(a+3,MAXA)))]
    print(f'  {k}: 위험이 이후 평균의 1.2배 안으로 들어와 3년 유지되는 첫 연차 = {fl[0] if fl else "없음"} (그 뒤 평균 {h.loc[fl[0]:].mean()*100:.1f}%)' if fl else f'  {k}: 없음')

print('\n[3] 계단 모형 꺾임점 (연차 0–14, 점포-년 가중 제곱오차 최소)')
def steps(L,k):
    M=min(MAXA,int((L.점포년>=200).sum())); h=L.위험.values[:M]; w=L.점포년.values[:M]; best=None
    for cut in itertools.combinations(range(1,M),k):
        b=[0,*cut,M]; sse=sum((w[i:j]*(h[i:j]-np.average(h[i:j],weights=w[i:j]))**2).sum() for i,j in zip(b,b[1:]))
        if best is None or sse<best[0]: best=(sse,cut,[np.average(h[i:j],weights=w[i:j]) for i,j in zip(b,b[1:])])
    return best
for k in ['전체','개인','프랜차이즈','2005–14 개업','2015–23 개업']:
    out=[]
    for n in (1,2,3):
        sse,cut,lv=steps(T[k],n); out.append(f'{n}개: 꺾임 {list(cut)}년, 구간 위험 '+'/'.join(f'{x*100:.1f}' for x in lv))
    print(f'  {k}: '+' | '.join(out))

print('\n[4] 조건별 연차 위험 (%, 2021–26 관측 구간만: 상권 매출이 있는 기간)')
W=pd.read_csv(f'{P}/quadrant_trdar.csv',dtype={'상권코드':str}).set_index('상권코드')
W['매출4']=pd.qcut(W.점포당매출_앞,4,labels=['매출 하위25%','25–50%','50–75%','매출 상위25%'])
c2=c.join(W.매출4,on='상권코드'); W0=pd.Timestamp('2024-01-01')   # 매출은 2021–23로 나누고 폐업은 2024 이후로 봄(앞뒤 분리)
rows={}
for k,g in c2.dropna(subset=['매출4']).groupby('매출4'):
    L=life(g); rows[k]=[L.loc[a:b].폐업.sum()/L.loc[a:b].점포년.sum()*100 for a,b in [(0,0),(1,2),(3,4),(5,9),(10,15)]]
print(pd.DataFrame(rows,index=['첫해','2–3년차','4–5년차','6–10년차','11년차 이상']).round(1).to_string())
g=c[c.프랜차이즈]; br=bf.f.groupby('브랜드').y.mean(); top=set(br[br<=br.median()].index)
rows={}
for nm,x in [('개인',c[~c.프랜차이즈]),('폐업 낮은 브랜드',g[g.브랜드.isin(top)]),('폐업 높은 브랜드',g[~g.브랜드.isin(top)&g.브랜드.isin(br.index)])]:
    L=life(x); rows[nm]=[L.loc[a:b].폐업.sum()/L.loc[a:b].점포년.sum()*100 for a,b in [(0,0),(1,2),(3,4),(5,9),(10,15)]]
print('  (2024–26 관측) 브랜드 묶음별:'); print(pd.DataFrame(rows,index=['첫해','2–3년차','4–5년차','6–10년차','11년차 이상']).round(1).to_string())

# 그림
plt.rcParams.update({'font.family':'AppleGothic','axes.unicode_minus':False}); INK,SUB,GRID,BG='#0b0b0b','#52514e','#e4e3df','#fcfcfb'
fig,ax=plt.subplots(figsize=(10,5.4),facecolor=BG); ax.set_facecolor(BG)
for k,col in [('개인','#eb6834'),('프랜차이즈','#2a78d6')]:
    L=T[k].loc[:12]; ax.fill_between(L.index+1,L.하한*100,L.상한*100,color=col,alpha=.12,lw=0); ax.plot(L.index+1,L.위험*100,color=col,lw=2,marker='o',ms=5)
    ax.text(13.2,L.위험.iloc[-1]*100,k,color=INK,va='center',fontsize=11)
ax.set_xticks(range(1,14)); ax.set_xlabel('영업 연차 (년째)',color=SUB); ax.set_ylabel('그해 실제 폐업 위험 (%)',color=SUB)
ax.set_title('치킨집 영업 연차별 폐업 위험',loc='left',color=INK,fontsize=14); ax.grid(axis='y',color=GRID,lw=.8); ax.set_xlim(.5,14.3)
ax.tick_params(colors=SUB,length=0); [s.set_visible(False) for s in ax.spines.values()]
fig.text(.01,.01,'서울 인허가 치킨집(B_넓음) 2010-01~2026-11 관측 점포-년. 실제 폐업 = 같은 자리 재등록 제외. 띠 = 95% 구간.',fontsize=8.5,color=SUB)
fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(f'{O}/figures/survival_by_age.png',dpi=170,facecolor=BG)
pd.concat({k:v for k,v in T.items()},axis=1).round(4).to_csv(f'{P}/survival_by_age.csv',encoding='utf-8-sig')
