# 매출(인기) × 폐업 사분면: 폐업률만이 아니라 '잘 팔리면서 안 망하는' 곳을 찾고, 그 분류가 다음 기간에도 유지되는지 검증
#  매출: 서울시 상권분석서비스 상권 단위 치킨전문점 추정매출·점포(trdar_fetch.py, 2021Q1–2026Q2) → 점포당 연매출, 점포당 결제 건수
#  폐업: 인허가 치킨집(chicken_wide.py B_넓음)을 상권 폴리곤에 배정, 기간 안 실제 폐업(같은 자리 재등록 제외) / 영업 점포-년. 상권별 수축(점포-년 10)
#  기간: 앞(2021–23)으로 사분면을 나누고 → 뒤(2024–2026H1)의 폐업·매출로 확인
#  [1] 상권  [2] 구·상권 유형·사람유형  [3] 브랜드(공정위 평균매출 × 서울 3년 폐업, 앞 2017–20 / 뒤 2021–24)
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from scipy.stats import spearmanr
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/상권_분기'; P=f'{O}/data/processed'
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
bf=load('brand_fit'); c=bf.c.copy(); A=bf.aa.A
c['상권코드']=np.where(bf.gi>=0,A.상권코드.astype(str).values[np.clip(bf.gi,0,None)],None)
PER={'앞':('2021-01-01','2024-01-01'),'뒤':('2024-01-01','2026-07-01')}
def close_stats(g,a,b):
    a,b=pd.Timestamp(a),pd.Timestamp(b); cl=g.close.fillna(pd.Timestamp('2100-01-01'))
    yrs=((np.minimum(cl,b)-np.maximum(g.open,a)).dt.days.clip(lower=0))/365.25
    k=((cl>=a)&(cl<b)&~g.재등록).sum(); return yrs.sum(),k
s=pd.read_csv(f'{R}/치킨_매출_상권.csv'); t=pd.read_csv(f'{R}/치킨_점포_상권.csv')
m=s.merge(t[['STDR_YYQU_CD','TRDAR_CD','SIMILR_INDUTY_STOR_CO']],on=['STDR_YYQU_CD','TRDAR_CD']); m['상권코드']=m.TRDAR_CD.astype(str)
m['기간']=np.where(m.STDR_YYQU_CD<20241,'앞','뒤')
def sales(g):   # 점포당 연매출(만원), 점포당 연 결제 건수
    n=g.SIMILR_INDUTY_STOR_CO.sum(); return pd.Series({'점포당매출':g.THSMON_SELNG_AMT.sum()/n*4/1e4,'점포당건수':g.THSMON_SELNG_CO.sum()/n*4,'점포분기':n})
def table(key):
    rows=[]
    for p,(a,b) in PER.items():
        sv=m.groupby(['기간',key] if key in m else ['기간','상권코드']).apply(sales) if key in m else None
        cs=c[c[key].notna()].groupby(key).apply(lambda g:pd.Series(dict(zip(['점포년','폐업'],close_stats(g,a,b)))))
        cs['기간']=p; rows.append(cs)
    return pd.concat(rows).reset_index().set_index(['기간',key])
p0={}
# ---------- 1. 상권 ----------
S=m.groupby(['기간','상권코드']).apply(sales); C=table('상권코드'); T=S.join(C,how='inner').reset_index()
for p in PER: q=T[T.기간==p]; p0[p]=q.폐업.sum()/q.점포년.sum()
T['폐업률']=(T.폐업+10*T.기간.map(p0))/(T.점포년+10)   # 연 폐업률, 수축
W=T.pivot(index='상권코드',columns='기간',values=['점포당매출','점포당건수','폐업률','점포년','폐업','점포분기'])
W=W[(W[('점포년','앞')]>=5)&(W[('점포분기','앞')]>=8)&W[('점포당매출','뒤')].notna()].copy()
W.columns=[f'{a}_{b}' for a,b in W.columns]; W=W.join(A.set_index(A.상권코드.astype(str))[['상권명','상권유형','구명']])
print(f'[1] 상권: 매출 자료와 치킨집이 함께 있는 상권 {len(W)}개 (앞 2021–23 점포-년 5 이상). 연 실제 폐업률 앞 {p0["앞"]:.3f}, 뒤 {p0["뒤"]:.3f}')
r=spearmanr(W.점포당매출_앞,W.폐업률_앞); print(f'  매출↔폐업 (앞 기간, 상권 단위) 순위상관 {r.statistic:+.2f} (p={r.pvalue:.3f}); 건수↔폐업 {spearmanr(W.점포당건수_앞,W.폐업률_앞).statistic:+.2f}')
W['매출구간']=pd.qcut(W.점포당매출_앞,4,labels=['매출 하위25%','25–50%','50–75%','매출 상위25%'])
G4=W.groupby('매출구간').agg(상권=('폐업_앞','size'),점포년=('점포년_앞','sum'),폐업=('폐업_앞','sum'),점포년뒤=('점포년_뒤','sum'),폐업뒤=('폐업_뒤','sum'))
print('  점포당 매출 4분위별 연 폐업률 (앞 / 뒤):',', '.join(f'{k} {r.폐업/r.점포년:.3f} / {r.폐업뒤/r.점포년뒤:.3f}' for k,r in G4.iterrows()))
for v in ['점포당매출','점포당건수','폐업률']: print(f'  지속성 {v} 앞↔뒤 순위상관 {spearmanr(W[v+"_앞"],W[v+"_뒤"]).statistic:+.2f}')
mx,my=W.점포당매출_앞.median(),W.폐업률_앞.median()
W['사분면']=np.select([(W.점포당매출_앞>=mx)&(W.폐업률_앞<my),(W.점포당매출_앞>=mx)&(W.폐업률_앞>=my),(W.점포당매출_앞<mx)&(W.폐업률_앞<my)],['① 잘 팔리고 안 망함','② 잘 팔리지만 많이 망함','③ 덜 팔리지만 안 망함'],'④ 덜 팔리고 많이 망함')
Q=W.groupby('사분면').agg(상권=('상권명','size'),앞_매출=('점포당매출_앞','median'),뒤_매출=('점포당매출_뒤','median'),앞_점포년=('점포년_앞','sum'),앞_폐업=('폐업_앞','sum'),뒤_점포년=('점포년_뒤','sum'),뒤_폐업=('폐업_뒤','sum'))
Q['앞_폐업률']=Q.앞_폐업/Q.앞_점포년; Q['뒤_폐업률']=Q.뒤_폐업/Q.뒤_점포년
print('  사분면(앞 기간 중앙값 기준) → 뒤 기간 확인 (매출은 점포당 연매출 중앙값, 만원; 폐업률은 점포-년 기준 연율)')
print(Q[['상권','앞_매출','뒤_매출','앞_폐업률','뒤_폐업률','뒤_점포년']].round(3).to_string())
W['유지']=np.select([(W.점포당매출_뒤>=W.점포당매출_뒤.median())&(W.폐업률_뒤<W.폐업률_뒤.median())],[1],0)
print('  ①로 분류된 상권 중 뒤 기간에도 ①인 비율',f'{W[W.사분면.str.startswith("①")].유지.mean():.2f}','(무작위면 약 0.25)')
print('  ① 상권 유형 구성:',W[W.사분면.str.startswith('①')].상권유형.value_counts().to_dict(),'| 전체:',W.상권유형.value_counts().to_dict())
top=W[W.사분면.str.startswith('①')&(W.유지==1)].sort_values('점포년_앞',ascending=False)
print('  두 기간 모두 ①인 상권(점포-년 많은 순 10):',', '.join(f'{r.구명} {r.상권명}' for _,r in top.head(10).iterrows()))
W.reset_index().round(4).to_csv(f'{P}/quadrant_trdar.csv',index=False,encoding='utf-8-sig')

# ---------- 2. 구·유형 ----------
c['구명']=np.where(bf.gi>=0,A.구명.values[np.clip(bf.gi,0,None)],None); c['상권유형2']=np.where(bf.gi>=0,A.상권유형.values[np.clip(bf.gi,0,None)],None)
m=m.merge(A[['상권코드','구명','상권유형']].assign(상권코드=A.상권코드.astype(str)).rename(columns={'상권유형':'상권유형2'}),on='상권코드',how='left')
T2=pd.read_csv(f'{P}/area_types_people.csv').set_index('TRDAR_CD').유형.map({0:'직장가',1:'주거지',2:'먹자골목'}); m['사람유형']=m.TRDAR_CD.map(T2)
print('\n[2] 묶음별 (앞 → 뒤): 점포당 연매출(만원), 연 폐업률')
for key in ['상권유형2','사람유형','구명']:
    S2=m.groupby(['기간',key]).apply(sales); C2=table(key); X=S2.join(C2,how='inner'); X['폐업률']=X.폐업/X.점포년
    X=X.reset_index().pivot(index=key,columns='기간',values=['점포당매출','폐업률','점포년']); X.columns=[f'{a}_{b}' for a,b in X.columns]
    X=X[X.점포년_앞>=30]
    if key=='구명':
        mx2,my2=X.점포당매출_앞.median(),X.폐업률_앞.median()
        X['사분면']=np.select([(X.점포당매출_앞>=mx2)&(X.폐업률_앞<my2),(X.점포당매출_앞>=mx2)],['①','②'],np.where(X.폐업률_앞<my2,'③','④'))
        X['사분면_뒤']=np.select([(X.점포당매출_뒤>=X.점포당매출_뒤.median())&(X.폐업률_뒤<X.폐업률_뒤.median()),(X.점포당매출_뒤>=X.점포당매출_뒤.median())],['①','②'],np.where(X.폐업률_뒤<X.폐업률_뒤.median(),'③','④'))
        print(f'  구 {len(X)}개: 매출↔폐업 순위상관 앞 {spearmanr(X.점포당매출_앞,X.폐업률_앞).statistic:+.2f}, 매출 지속성 {spearmanr(X.점포당매출_앞,X.점포당매출_뒤).statistic:+.2f}, 폐업 지속성 {spearmanr(X.폐업률_앞,X.폐업률_뒤).statistic:+.2f}, 사분면 유지 {(X.사분면==X.사분면_뒤).mean():.2f}')
        print('   앞·뒤 모두 ①:',', '.join(X[(X.사분면=='①')&(X.사분면_뒤=='①')].index),'| 모두 ④:',', '.join(X[(X.사분면=='④')&(X.사분면_뒤=='④')].index))
        X.round(4).to_csv(f'{P}/quadrant_gu.csv',encoding='utf-8-sig')
    else: print(f'  {key}:'); print(X[['점포당매출_앞','점포당매출_뒤','폐업률_앞','폐업률_뒤']].round(3).to_string())

# ---------- 3. 브랜드 ----------
bt=pd.read_csv(f'{P}/chicken_brand_table.csv')[['브랜드','ftc']].dropna(); ft=pd.read_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv')
ft=ft[ft.brandNm.isin(bt.ftc)].groupby(['brandNm','yr']).agg(매출=('avrgSlsAmt','max'),가맹점=('frcsCnt','max'),종료=('ctrtEndCnt','max'),해지=('ctrtCncltnCnt','max')).reset_index()
ft['이탈률']=(ft.종료.fillna(0)+ft.해지.fillna(0))/ft.가맹점
c['y3']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
rows=[]
for p,(sy,oy) in {'앞':((2017,2020),(2015,2018)),'뒤':((2021,2024),(2019,2022))}.items():
    f=ft[ft.yr.between(*sy)].groupby('brandNm').agg(매출=('매출','mean'),이탈률=('이탈률','mean'))
    y=c[c.프랜차이즈&c.연.between(*oy)].groupby('브랜드').y3.agg(['mean','size'])
    x=bt.set_index('ftc').join(f).set_index('브랜드').join(y); x['기간']=p; rows.append(x)
Bd=pd.concat(rows).reset_index().pivot(index='브랜드',columns='기간',values=['매출','mean','size','이탈률']); Bd.columns=[f'{a}_{b}' for a,b in Bd.columns]
Bd=Bd[(Bd.size_앞>=15)&(Bd.size_뒤>=15)&Bd.매출_앞.notna()&Bd.매출_뒤.notna()].copy(); Bd[['매출_앞','매출_뒤']]/=1e5
print(f'\n[3] 브랜드 {len(Bd)}개: 공정위 평균매출(억, 전국) × 서울 3년 실제 폐업. 앞 = 매출 2017–20·개업 2015–18, 뒤 = 매출 2021–24·개업 2019–22')
print(f'  매출↔폐업 순위상관 앞 {spearmanr(Bd.매출_앞,Bd.mean_앞).statistic:+.2f}, 뒤 {spearmanr(Bd.매출_뒤,Bd.mean_뒤).statistic:+.2f}; 지속성 매출 {spearmanr(Bd.매출_앞,Bd.매출_뒤).statistic:+.2f}, 폐업 {spearmanr(Bd.mean_앞,Bd.mean_뒤).statistic:+.2f}')
mx3,my3=Bd.매출_앞.median(),Bd.mean_앞.median(); mx4,my4=Bd.매출_뒤.median(),Bd.mean_뒤.median()
q=lambda s,y,a,b:np.select([(s>=a)&(y<=b),(s>=a)],['①','②'],np.where(y<=b,'③','④'))
Bd['사분면_앞']=q(Bd.매출_앞,Bd.mean_앞,mx3,my3); Bd['사분면_뒤']=q(Bd.매출_뒤,Bd.mean_뒤,mx4,my4)
Bd=Bd.rename(columns={'mean_앞':'폐업_앞','mean_뒤':'폐업_뒤','size_앞':'n_앞','size_뒤':'n_뒤'})
print(Bd[['매출_앞','폐업_앞','n_앞','사분면_앞','매출_뒤','폐업_뒤','n_뒤','사분면_뒤','이탈률_뒤']].sort_values(['사분면_앞','폐업_앞']).round(3).to_string())
print(f'  사분면 유지 {(Bd.사분면_앞==Bd.사분면_뒤).mean():.2f}')
Bd.round(4).to_csv(f'{P}/quadrant_brand.csv',encoding='utf-8-sig')

# ---------- 그림: 상권 사분면(앞) + 뒤 기간 확인 ----------
plt.rcParams.update({'font.family':'AppleGothic','axes.unicode_minus':False})
INK,SUB,GRID,BG='#0b0b0b','#52514e','#e4e3df','#fcfcfb'; COL={'①':'#2a78d6','②':'#eda100','③':'#1baf7a','④':'#e34948'}
fig,ax=plt.subplots(1,2,figsize=(13,5.6),facecolor=BG,gridspec_kw={'width_ratios':[1.5,1]})
a=ax[0]; a.set_facecolor(BG)
for k,g in W.groupby(W.사분면.str[0]): a.scatter(g.점포당매출_앞/1e4,g.폐업률_앞*100,s=np.clip(g.점포년_앞,5,120),color=COL[k],alpha=.75,edgecolor=BG,linewidth=1,label=Q.index[[i for i,x in enumerate(Q.index) if x.startswith(k)][0]])
a.axvline(mx/1e4,color=GRID,lw=1,zorder=0); a.axhline(my*100,color=GRID,lw=1,zorder=0); a.set_xscale('log')
a.set_xlabel('치킨집 점포당 연매출 (억 원, 로그 눈금, 2021–23)',color=SUB); a.set_ylabel('연 실제 폐업률 (%, 2021–23, 수축)',color=SUB); a.set_title(f'서울 상권 {len(W)}곳: 매출 × 폐업',loc='left',color=INK,fontsize=13)
a.legend(frameon=False,fontsize=9,loc='upper right'); a.tick_params(colors=SUB,length=0); [x.set_visible(False) for x in a.spines.values()]
from matplotlib.ticker import FuncFormatter,FixedLocator
a.xaxis.set_major_locator(FixedLocator([.1,.2,.5,1,2,5,10])); a.xaxis.set_minor_locator(FixedLocator([])); a.xaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}'))
b=ax[1]; b.set_facecolor(BG); y0=np.arange(len(G4)); BL=['#b7d3f6','#86b6ef','#3987e5','#1c5cab']
v=(G4.폐업뒤/G4.점포년뒤*100).values; b.barh(y0,v,height=.6,color=BL)
for i,x in enumerate(v): b.text(x+.2,i,f'{x:.1f}%',va='center',fontsize=10,color=INK)
b.set_yticks(y0,[str(k) for k in G4.index],color=INK); b.invert_yaxis(); b.set_xlabel('뒤 기간(2024–26H1) 연 실제 폐업률 (%)',color=SUB)
b.set_title('앞 기간 매출이 높던 상권은 뒤에 덜 망함\n',loc='left',color=INK,fontsize=13); b.text(0,1.015,'세로: 앞 기간(2021–23) 점포당 매출 4분위',transform=b.transAxes,color=SUB,fontsize=9)
b.tick_params(colors=SUB,length=0); [x.set_visible(False) for x in b.spines.values()]
fig.text(.01,.01,'매출: 서울시 상권분석서비스 추정매출(카드 기반, 상권 단위). 폐업: 서울 인허가 치킨집, 같은 자리 재등록 제외. 점 크기 = 영업 점포-년.',fontsize=8.5,color=SUB)
fig.tight_layout(rect=(0,.04,1,1)); fig.savefig(f'{O}/figures/quadrant_trdar.png',dpi=170,facecolor=BG)
