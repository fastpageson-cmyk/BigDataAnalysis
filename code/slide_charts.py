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
MID='#d97706'
# DECK=1: 발표 덱용(남색·파랑 브랜드 색, 흰 바탕, 그림 안 제목·각주 없음 → 제목·출처는 장표가 맡음) → figures/deck/
DECK=os.environ.get('DECK')=='1'
if DECK: INK,SUB,GRID,BG,ACC,MUT,MID='#051C2C','#5A6872','#D9DEE3','#FFFFFF','#2251FF','#B8C2CA','#7C93FF'
OUT=f'{O}/figures/deck' if DECK else f'{O}/figures'; os.makedirs(OUT,exist_ok=True)
def save(fig,name):
    if DECK:
        for t in list(fig.texts): t.remove()
        if fig._suptitle is not None: fig._suptitle.remove(); fig._suptitle=None
        if len(fig.axes)==1: fig.axes[0].set_title(''); fig.axes[0].set_title('',loc='left')
        if fig.legends:
            fig.tight_layout(rect=(0,0,1,.9))
            for lg in fig.legends: lg.set_bbox_to_anchor((0.015,0.995))
        else: fig.tight_layout()
    fig.savefig(f'{OUT}/{name}',dpi=170,facecolor=BG); plt.close(fig)

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
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart1_competition_change.png')
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
fig.text(.02,.01,'판단 시점에 영업 중인 서울 치킨집. 상권은 판단 시점 이전 매출로만 나눴다(결과를 보기 전에 나눔). 세로선 = 95% 구간. \n폐업은 같은 자리 90일 안 재등록 제외. 출처: 서울시 인허가, 상권분석서비스.',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart3_contrast.png')

# ── 발표 보조 그림 3장 (11/12 추가) ──
# ④ 업종별 동네 효과: industry_location.csv(industry_location.py), 업종 10개만
I=pd.read_csv(f'{P}/industry_location.csv'); I=I[~I.업종.str.contains('·')].sort_values('AUC_동네')
fig,ax=plt.subplots(figsize=(8,4.8),facecolor=BG); ax.set_facecolor(BG)
cols=[ACC if u=='치킨' else MUT for u in I.업종]; ax.barh(I.업종,I.AUC_동네,color=cols,height=.62)
for y_,(u,v) in enumerate(zip(I.업종,I.AUC_동네)): ax.text(v+.002,y_,f'{v:.3f}',va='center',fontsize=9.5,color=INK)
ax.axvline(.5,color=INK,lw=1,ls='--'); ax.text(.5,len(I)-.4,' 동전 던지기(0.5)',fontsize=9,color=SUB,va='bottom')
ax.set_xlim(.45,.6); ax.set_xlabel('동네 과거 폐업률로 새 가게 3년 폐업을 맞히는 힘 (AUC, 0.5 = 동전 던지기)',color=SUB)
ax.set_title('치킨은 동네를 알아도 폐업을 못 맞힌다 (음식점 업종 중 가장 약함)',loc='left',color=INK,fontsize=13,pad=10)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors=SUB); ax.grid(axis='x',color=GRID,lw=.8); ax.set_axisbelow(True)
fig.text(.02,.01,'서울 인허가 2018–22 개업, 500m 칸의 2010–14 개업 전 업종 3년 실제 폐업률. 점주 교체(같은 자리 90일 안 재등록) 제외,\n브랜드·면적·연도는 업종마다 같게 통제. 출처: 서울시 일반·휴게음식점 인허가. 스크립트 industry_location.py',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart_industry.png')

# ⑤ 기준선 비교: model_eval.csv(model.py), 평가 2021–22 개업 965곳
E=pd.read_csv(f'{P}/model_eval.csv').set_index('모델')
rows=[('⓪ 전원 생존(0)','모두 산다고 찍기'),('② 프랜차이즈 평균','프랜차이즈냐 개인이냐'),('③ 브랜드 과거 폐업률','브랜드 과거 폐업률 (기준선)'),('④ 브랜드+면적','+ 가게 크기'),('★ 보조지표 방정식','최종: 가게 크기 + 브랜드 서울 영업 기간')]
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4.6),facecolor=BG,gridspec_kw={'width_ratios':[1.5,1]})
for a in (a1,a2): a.set_facecolor(BG); [s.set_visible(False) for s in a.spines.values()]; a.tick_params(colors=SUB)
labs=[l for _,l in rows]; vals=[E.loc[k,'AUC'] for k,_ in rows]; cl=[ACC if k.startswith('★') else (INK if k.startswith('③') else MUT) for k,_ in rows]
a1.barh(labs[::-1],vals[::-1],color=cl[::-1],height=.6)
for y_,v in enumerate(vals[::-1]): a1.text(v+.004,y_,f'{v:.3f}',va='center',fontsize=9.5,color=INK)
a1.set_xlim(.45,.76); a1.axvline(.5,color=SUB,lw=1,ls='--'); a1.set_title('맞히는 힘 (AUC)',loc='left',color=INK,fontsize=11)
k1,k2='③ 브랜드 과거 폐업률','★ 보조지표 방정식'
gx=np.arange(2); w=.36
for j,(k,lab,c_) in enumerate([(k1,'기준선',INK),(k2,'최종 모델',ACC)]):
    v=[E.loc[k,'상위20pct_폐업률']*100,E.loc[k,'하위20pct_폐업률']*100]; a2.bar(gx+(j-.5)*w,v,w,color=c_,label=lab)
    for x_,vv in zip(gx+(j-.5)*w,v): a2.text(x_,vv+1,f'{vv:.0f}%',ha='center',fontsize=9.5,color=INK)
a2.set_xticks(gx); a2.set_xticklabels(['위험하다고 본\n상위 20%','안전하다고 본\n하위 20%'],color=INK); a2.set_ylim(0,62)
a2.set_title('실제로 3년 안에 닫은 비율',loc='left',color=INK,fontsize=11); a2.legend(frameon=False,fontsize=9)
fig.suptitle('가게 크기와 브랜드 정보만으로 기준선을 넘었다',x=.02,ha='left',fontsize=13,color=INK)
fig.text(.02,.01,'학습 2018–20 개업 1,804곳 → 평가 2021–22 개업 965곳(평가 기간은 한 번만 봄).\n기준선 대비 +%.3f (95%% %+.3f~%+.3f). 출처: 서울시 인허가, 공정위 가맹정보. 스크립트 model.py'%(E.loc[k2,'AUC차_vs브랜드'],E.loc[k2,'CI하'],E.loc[k2,'CI상']),fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart_model.png')

# ⑥ 이탈률 계단: brand_signal.py의 4분위표(가맹점 가중)
sp=importlib.util.spec_from_file_location('bs',f'{O}/code/brand_signal.py'); bs=importlib.util.module_from_spec(sp)
with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(bs)
q=bs.T.iloc[-4:]; v=(q.다음해이탈률_가중*100).values; m_=(q.신호평균*100).values
fig,ax=plt.subplots(figsize=(8,4.6),facecolor=BG); ax.set_facecolor(BG)
ax.bar(range(4),v,color=[MUT,MUT,MID,ACC],width=.6)
for i_,vv in enumerate(v): ax.text(i_,vv+.6,f'{vv:.1f}%',ha='center',fontsize=11,color=INK)
ax.set_xticks(range(4)); ax.set_xticklabels([f'{n}\n(올해 평균 {mm:.1f}%)' for n,mm in zip(['올해 이탈 낮음','','','올해 이탈 높음'],m_)],color=INK,fontsize=9.5)
ax.set_ylabel('다음 해 가맹점 이탈률 (%)',color=SUB); ax.set_ylim(0,max(v)*1.2)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors=SUB); ax.grid(axis='y',color=GRID,lw=.8); ax.set_axisbelow(True)
ax.set_title('올해 가맹점이 많이 떠난 브랜드는 내년에도 많이 떠난다',loc='left',color=INK,fontsize=13,pad=10)
fig.text(.02,.01,'치킨 브랜드 143개(연초 가맹점 30곳 이상), 브랜드-연도를 올해 이탈률(계약종료+해지÷연초 가맹점)로 4등분. 다음 해 이탈률은 가맹점 수 가중.\n출처: 공정위 가맹사업 정보공개서(가맹점 현황 15110241). 스크립트 brand_signal.py',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart_exit.png')
print('보조 그림: 업종',I[['업종','AUC_동네']].round(3).values.tolist(),' 이탈 4분위',np.round(v,1).tolist())

# ② 발표용 브랜드 산점도(한 장): x = 서울 3년 실제 폐업률(95% 구간선), y = 공정위 2024 전국 가맹점 평균매출, 원 크기 = 창업비
#    입력 brand_risk_return.csv(brand_risk_return.py). 원본 두 장짜리 그림은 라벨이 겹쳐서 발표용은 라벨 위치를 손으로 정함
B=pd.read_csv(f'{P}/brand_risk_return.csv'); HL={'교촌','60계','호식이','멕시카나'}
OFF={'굽네':(-9,7,'right'),'푸라닭':(-9,-7,'right'),'지코바':(8,-6,'left'),'처갓집':(8,5,'left')}
fig,ax=plt.subplots(figsize=(9,5.6),facecolor=BG); ax.set_facecolor(BG)
for _,r in B.iterrows():
    c_=ACC if r.브랜드 in HL else MUT
    ax.hlines(r.매출_억,r.lo,r.hi,color=c_,alpha=.35,lw=2,zorder=1); ax.scatter(r.폐업,r.매출_억,s=r.창업비_억*170,color=c_,edgecolor=BG,lw=1.5,zorder=2)
    dx,dy,ha=OFF.get(r.브랜드,(9,0,'left')); lab=r.브랜드+(f' (창업비 {r.창업비_억:.1f}억)' if r.브랜드 in HL else '')
    ax.annotate(lab,(r.폐업,r.매출_억),xytext=(dx,dy),textcoords='offset points',ha=ha,va='center',fontsize=10 if r.브랜드 in HL else 9,color=INK)
ax.set_xlim(-1,52); ax.set_ylim(.8,7.6)
ax.set_xlabel('서울 3년 실제 폐업률 (%, 가로선은 95% 구간)',color=SUB); ax.set_ylabel('가맹점 연 평균매출 (억 원, 2024 전국)',color=SUB)
ax.text(.99,.98,'원 크기 = 창업비',transform=ax.transAxes,ha='right',va='top',fontsize=9,color=SUB)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors=SUB); ax.grid(color=GRID,lw=.8); ax.set_axisbelow(True)
ax.set_title('같은 프랜차이즈라도 브랜드마다 폐업률이 30배 넘게 다르고, 비싸다고 안전하지 않다',loc='left',color=INK,fontsize=12,pad=10)
fig.text(.02,.01,'서울 개업 25곳 이상이고 공정위 매출·창업비가 있는 14개 브랜드. 폐업률: 서울 인허가 2015–22 개업, 같은 자리 90일 안 재등록 제외.\n매출·창업비: 공정위 가맹사업 정보공개 2024 전국 평균(창업비는 임차보증금 제외, 매출은 이익이 아님). 스크립트 brand_risk_return.py, slide_charts.py',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart_brand.png')

# ⑦ 신규 개점 1곳당 기존 점포 인수(명의변경): brand_entry.csv(brand_signal.py), 공정위 2024 등록(2023 실적)
N=pd.read_csv(f'{P}/brand_entry.csv'); N['인수배']=N.nmChgCnt/N.newFrcsRgsCnt; N=N.sort_values('인수배')
fig,ax=plt.subplots(figsize=(8,5.4),facecolor=BG); ax.set_facecolor(BG)
cl=[ACC if b in ('교촌','지코바','60계') else MUT for b in N.브랜드]; ax.barh(N.브랜드,N.인수배,color=cl,height=.62)
for y_,(v,c3) in enumerate(zip(N.인수배,N.서울3년실제폐업)): ax.text(v+.15,y_,f'{v:.1f}곳   (서울 3년 폐업 {c3*100:.1f}%)',va='center',fontsize=8.8,color=INK)
ax.axvline(1,color=INK,lw=1,ls='--'); ax.text(1.05,-.9,'1 = 신규와 인수가 같음',fontsize=8.5,color=SUB)
ax.set_xlim(0,15); ax.set_xlabel('신규 개점 1곳당 기존 가맹점 인수(명의변경) 수',color=SUB)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(colors=SUB); ax.grid(axis='x',color=GRID,lw=.8); ax.set_axisbelow(True)
ax.set_title('폐업이 적은 교촌·지코바·60계는 새로 열기보다 인수로 들어간다',loc='left',color=INK,fontsize=12,pad=10)
fig.text(.02,.01,'공정위 가맹점 현황(15110241) 2024 등록분: 신규 개점 수, 명의변경 수. 가맹점 100곳 이상 치킨 브랜드 54곳 합계는 신규 2,077 vs 인수 2,263.\n폐업률과 이 비율의 순위상관 0.42(p=0.14)로 방향만 같음: 규칙이 아니라 사례로 읽는다. 스크립트 brand_signal.py, slide_charts.py',fontsize=8,color=SUB)
fig.tight_layout(rect=(0,.07,1,.95)); save(fig,'chart_entry.png')
