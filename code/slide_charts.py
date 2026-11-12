# 발표 차트 ①·③ 그리기 (②는 brand_risk_return.py). 값과 검증은 chart_check.py와 같다.
#  ① 경쟁 밀도 '변화' 산점도: 행정동-연, x = 치킨집 수 전년 대비 변화, y = 점포당 매출 전년 대비 변화(로그 차이), 점포 수 가중 회귀선
#     입력 data/processed/dong_chicken_panel.csv (dong_market.py)
#  ③ 대조 막대: 상권 점포당 매출 상위 25% vs 나머지, 두 시점(A 2023-12-31, B 2021-12-31) × 개인/프랜차이즈, 95% Wilson 구간
#     chart_check.py의 a1, a2 표본을 그대로 씀(몇 분 걸림)
# 출력 figures/chart1_competition_change.png, figures/chart3_contrast.png, data/processed/chart3_contrast.csv
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
plt.rcParams.update({'font.family':'AppleGothic','axes.unicode_minus':False})
INK,SUB,GRID,BG,ACC,MUT='#0b0b0b','#52514e','#e4e3df','#fcfcfb','#c2410c','#a8a29e'

# ① 경쟁 밀도 변화
p=pd.read_csv(f'{P}/dong_chicken_panel.csv',dtype={'동':str}).dropna(subset=['d매출','d점포'])
m=smf.wls('d매출~d점포+C(연)',p,weights=p.점포).fit(cov_type='cluster',cov_kwds={'groups':p.동.astype('category').cat.codes})
b,pv=m.params['d점포'],m.pvalues['d점포']
fig,ax=plt.subplots(figsize=(8,5.2),facecolor=BG); ax.set_facecolor(BG)
x,y=p.d점포*100,p.d매출*100
ax.scatter(x,y,s=np.clip(p.점포,2,40)*1.5,color=MUT,alpha=.35,lw=0)
xs=np.linspace(-60,60,50); ax.plot(xs,b*xs+np.average(y-b*x,weights=p.점포),color=ACC,lw=2.2)
ax.axhline(0,color=GRID,lw=1,zorder=0); ax.axvline(0,color=GRID,lw=1,zorder=0)
ax.set_xlim(-60,60); ax.set_ylim(-60,60)
ax.set_xlabel('동 치킨집 수 전년 대비 변화 (%)',color=SUB); ax.set_ylabel('동 치킨집 점포당 매출 전년 대비 변화 (%)',color=SUB)
ax.set_title(f'동에서 치킨집이 1% 늘면 점포당 매출은 {abs(b):.2f}% 줄었다',loc='left',color=INK,fontsize=13,pad=10)
ax.text(.99,.97,f'기울기 {b:+.2f} (p={pv:.1g}), 동-연 {len(p):,}개',transform=ax.transAxes,ha='right',va='top',color=ACC,fontsize=10)
for s in ax.spines.values(): s.set_color(GRID)
ax.tick_params(colors=SUB)
fig.text(.02,.01,'서울시 상권분석서비스(행정동) 2021–2025. 점 크기 = 전년 치킨집 수, 선 = 점포 수 가중 회귀(연도 통제, 동 군집 표준오차). 같은 동 안의 변화라\n"수요 많은 곳에 가게가 몰리는" 효과를 뺐다.',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.06,1,1)); fig.savefig(f'{O}/figures/chart1_competition_change.png',dpi=170,facecolor=BG); plt.close(fig)
print(f'① 기울기 {b:+.3f} p={pv:.2g} n={len(p)}')

# ③ 대조 막대
sp=importlib.util.spec_from_file_location('cc',f'{O}/code/chart_check.py'); cc=importlib.util.module_from_spec(sp)
with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(cc)
R=[]
for nm,lab,a in [('A','2023년 말 판단\n(매출 2021–23 → 폐업 2024–26)',cc.a1),('B','2021년 말 판단\n(매출 2021 → 폐업 2022–24)',cc.a2)]:
    for k,f in [('개인',0),('프랜차이즈',1)]:
        for top in (1,0):
            g=a[(a.프랜==f)&((a.pct>=.75)==bool(top))]; lo,hi=proportion_confint(g.y.sum(),len(g),method='wilson')
            R.append(dict(시점=nm,시점설명=lab,구분=k,상권='매출 상위 25%' if top else '나머지',n=len(g),폐업=int(g.y.sum()),폐업률=g.y.mean(),lo=lo,hi=hi))
R=pd.DataFrame(R); R.round(4).drop(columns='시점설명').to_csv(f'{P}/chart3_contrast.csv',index=False,encoding='utf-8-sig'); print(R.drop(columns='시점설명').round(3).to_string(index=False))
fig,axs=plt.subplots(1,2,figsize=(9,5),facecolor=BG,sharey=True)
for ax,(nm,g) in zip(axs,R.groupby('시점',sort=True)):
    ax.set_facecolor(BG)
    for i,(k,h) in enumerate(g.groupby('구분',sort=True)):
        for j,(_,r) in enumerate(h.iterrows()):
            xx=i+(j-.5)*.36; col=ACC if r.상권=='매출 상위 25%' else MUT
            ax.bar(xx,r.폐업률*100,.34,color=col); ax.errorbar(xx,r.폐업률*100,yerr=[[r.폐업률*100-r.lo*100],[r.hi*100-r.폐업률*100]],color=INK,lw=1,capsize=3)
            ax.text(xx,r.hi*100+.8,f'{r.폐업률*100:.1f}%',ha='center',color=INK,fontsize=9.5)
    ax.set_xticks([0,1]); ax.set_xticklabels(sorted(g.구분.unique()),color=INK,fontsize=11)
    ax.set_title(g.시점설명.iloc[0],color=INK,fontsize=10.5,loc='left')
    ax.grid(axis='y',color=GRID,lw=.8); ax.set_axisbelow(True); ax.set_ylim(0,36)
    for s in ax.spines.values(): s.set_visible(False)
    ax.tick_params(colors=SUB)
axs[0].set_ylabel('판단 뒤 약 3년 안 실제 폐업률 (%)',color=SUB)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=ACC,label='상권 점포당 매출 상위 25%'),Patch(color=MUT,label='나머지 상권')],frameon=False,loc='upper left',bbox_to_anchor=(.015,.935),ncol=2,fontsize=9.5)
fig.suptitle('매출이 높던 상권의 치킨집은 다음 기간에 덜 닫았다 (개인 치킨집에서)',x=.02,ha='left',fontsize=13,color=INK)
fig.text(.02,.01,'판단 시점에 영업 중인 서울 치킨집. 상권은 판단 시점 이전 매출로만 나눴다(결과를 보기 전에 나눔). 세로선 = 95% 구간. \n폐업은 같은 자리 1년 안 재등록 제외. 출처: 서울시 인허가, 상권분석서비스.',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.06,1,.87)); fig.savefig(f'{O}/figures/chart3_contrast.png',dpi=170,facecolor=BG)
