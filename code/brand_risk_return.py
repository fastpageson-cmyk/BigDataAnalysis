# 브랜드 수익·위험 지도: 가로 = 서울 3년 실제 폐업률(인허가, 2015–22 개업, 95% Wilson 구간), 세로 = 공정위 2024 전국 가맹점 평균매출 / 창업비 대비 매출
#  입력 data/processed/chicken_brand_table.csv (chicken_brand_table.py), 서울 개업 25곳 이상 + 공정위 매출·창업비가 있는 브랜드만
#  출력 figures/brand_risk_return.png, data/processed/brand_risk_return.csv
import os,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import spearmanr
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({'font.family':'AppleGothic','axes.unicode_minus':False})
t=pd.read_csv(f'{O}/data/processed/chicken_brand_table.csv')
t=t[(t.서울개업15_22>=25)&t.평균매출24_천원.notna()&t.창업비_천원.notna()].copy()
t['폐업']=t.서울3년실제폐업*100; k=(t.서울3년실제폐업*t.서울개업15_22).round()
lo,hi=proportion_confint(k,t.서울개업15_22,method='wilson'); t['lo']=lo*100; t['hi']=hi*100
t['매출_억']=t.평균매출24_천원/1e5; t['창업비_억']=t.창업비_천원/1e5; t['배수']=t['매출/창업비']
t[['브랜드','서울개업15_22','폐업','lo','hi','매출_억','창업비_억','배수']].round(2).to_csv(f'{O}/data/processed/brand_risk_return.csv',index=False,encoding='utf-8-sig')
print(t[['브랜드','서울개업15_22','폐업','lo','hi','매출_억','창업비_억','배수']].round(2).to_string(index=False))
for y in ['매출_억','배수','창업비_억']: r=spearmanr(t.폐업,t[y]); print(f'폐업률 vs {y}: Spearman {r.statistic:+.2f} (p={r.pvalue:.2f}, 브랜드 {len(t)}개)')
INK,SUB,GRID,BLUE,BG='#0b0b0b','#52514e','#e4e3df','#2a78d6','#fcfcfb'
fig,ax=plt.subplots(1,2,figsize=(13,6.2),facecolor=BG)
for a,(y,lab,ttl) in zip(ax,[('매출_억','가맹점 연 평균매출 (억 원, 2024 전국)','얼마나 파나'),('배수','연 평균매출 ÷ 창업비 (배)','들인 돈 대비 얼마나 파나')]):
    a.set_facecolor(BG); mx,my=t.폐업.median(),t[y].median()
    a.axvline(mx,color=GRID,lw=1,zorder=0); a.axhline(my,color=GRID,lw=1,zorder=0)
    a.hlines(t[y],t.lo,t.hi,color=BLUE,alpha=.25,lw=2,zorder=1)
    a.scatter(t.폐업,t[y],s=t.창업비_억*260,color=BLUE,edgecolor=BG,linewidth=2,zorder=2)
    a.set_xlim(-1.5,t.hi.max()+2); a.margins(y=.15); fig.canvas.draw(); placed=[]
    dots=[(X-R,Y-R,X+R,Y+R) for (X,Y),R in zip(a.transData.transform(t[['폐업',y]].values),np.sqrt(t.창업비_억*260)/2*fig.dpi/72)]
    for _,r in t.sort_values('창업비_억',ascending=False).iterrows():   # 겹치지 않는 자리(오른쪽→왼쪽→위→아래)를 차례로 시도
        px,py=a.transData.transform((r.폐업,r[y])); rad=np.sqrt(r.창업비_억*260)/2*fig.dpi/72; w,h=len(r.브랜드)*10.5*fig.dpi/72,13*fig.dpi/72; g=5*fig.dpi/72
        for dx,dy,ha,va in [(rad+g,0,'left','center'),(-rad-g,0,'right','center'),(0,rad+g,'center','bottom'),(0,-rad-g,'center','top'),(rad+g,h,'left','center'),(rad+g,-h,'left','center'),(-rad-g,h,'right','center'),(-rad-g,-h,'right','center')]:
            x0=px+dx-(0 if ha=='left' else w if ha=='right' else w/2); y0=py+dy-(h/2 if va=='center' else 0 if va=='bottom' else h); box=(x0,y0,x0+w,y0+h)
            if not any(box[0]<b[2] and b[0]<box[2] and box[1]<b[3] and b[1]<box[3] for b in placed+[d for d in dots if not (d[0]<px<d[2] and d[1]<py<d[3])]): break
        placed.append(box); a.annotate(r.브랜드,(r.폐업,r[y]),xytext=(dx*72/fig.dpi,dy*72/fig.dpi),textcoords='offset points',ha=ha,va=va,fontsize=10,color=INK)
    a.set_xlabel('서울 3년 실제 폐업률 (%, 가로선은 95% 구간)',color=SUB); a.set_ylabel(lab,color=SUB); a.set_title(ttl,loc='left',color=INK,fontsize=13,pad=10)
    a.tick_params(colors=SUB,length=0); [s.set_visible(False) for s in a.spines.values()]
    a.text(.995,.99,'← 덜 망함 · 위로 갈수록 많이 팜',transform=a.transAxes,color=SUB,fontsize=9,ha='right',va='top')
fig.suptitle('치킨 브랜드 수익·위험 지도',x=.06,ha='left',fontsize=16,color=INK,y=.98)
fig.text(.06,.015,'원 크기 = 창업비(임차보증금 제외). 회색 선 = 브랜드 중앙값. 폐업률: 서울 인허가 2015–22 개업(같은 자리 재등록 제외). 매출·창업비: 공정위 가맹사업 정보공개 2024, 전국 평균. 매출은 이익이 아님.',fontsize=8.5,color=SUB)
fig.tight_layout(rect=(.02,.04,1,.95)); fig.savefig(f'{O}/figures/brand_risk_return.png',dpi=170,facecolor=BG)
