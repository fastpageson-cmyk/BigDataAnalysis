# 발표 차트 3종 후보 검증: ① 경쟁 밀도 산점도 ② 브랜드 생존×매출 산점도 ③ 대조 검증(조건 통과 vs 아님, 결과 보기 전에 나눔)
#  ① 같은 시점 횡단면(상권별 치킨집 수 vs 점포당 매출·폐업)은 '수요가 많은 곳에 가게가 몰리는' 혼입이 있는지, 변화(동·상권 고정효과) 산점도는 포화를 보여주는지
#  ② 서울 3년 폐업(인허가) vs 공정위 전국 평균매출: 순위상관·구간 폭, 서울 폐업과 공정위 전국 이탈률이 같은 방향인지(출처가 달라도 버티나)
#  ③ '상권 점포당 매출 상위 25%' 대조를 두 번째 시점(2021-12-31, 2021년 매출만 사용 → 2022–24 폐업)에서 재현, 기준(상위 20·30·50%)을 바꿔도 유지, 상권 매출을 섞은 순위 1,000회와 비교
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.stats import spearmanr,pearsonr
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'; R=f'{O}/data/raw/상권_분기'
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
bf=load('brand_fit'); c=bf.c.copy(); A=bf.aa.A; FAR=pd.Timestamp('2100-01-01'); c['cc']=c.close.fillna(FAR)
c['상권코드']=np.where(bf.gi>=0,A.상권코드.astype(str).values[np.clip(bf.gi,0,None)],None)
s=pd.read_csv(f'{R}/치킨_매출_상권.csv'); t=pd.read_csv(f'{R}/치킨_점포_상권.csv')
m=s.merge(t[['STDR_YYQU_CD','TRDAR_CD','SIMILR_INDUTY_STOR_CO']],on=['STDR_YYQU_CD','TRDAR_CD']); m['상권코드']=m.TRDAR_CD.astype(str); m['연']=m.STDR_YYQU_CD//10
Y=m.groupby(['상권코드','연']).agg(매출=('THSMON_SELNG_AMT','sum'),점포분기=('SIMILR_INDUTY_STOR_CO','sum')).reset_index()
Y['점포']=Y.점포분기/4; Y['점포당매출']=Y.매출/Y.점포분기*4/1e8; Y=Y[Y.점포분기>0]

print('[① 경쟁 밀도 산점도]')
x=Y[(Y.연==2025)&(Y.점포>=1)]
r1=spearmanr(x.점포,x.점포당매출); print(f'  횡단면(2025, 상권 {len(x)}곳): 치킨집 수 vs 점포당 매출 순위상관 {r1.statistic:+.2f} (p={r1.pvalue:.3g}) → 많은 곳이 더 팔면 양(+), 포화가 보이려면 음(−)')
W=pd.read_csv(f'{P}/quadrant_trdar.csv',dtype={'상권코드':str})
r2=spearmanr(W.점포년_앞,W.폐업률_앞); print(f'  횡단면(2021–23, 상권 {len(W)}곳): 치킨집 점포-년 vs 연 폐업률 순위상관 {r2.statistic:+.2f} (p={r2.pvalue:.3g})')
Y=Y.sort_values(['상권코드','연']); g=Y.groupby('상권코드')
Y['d점포']=np.log(Y.점포)-np.log(g.점포.shift(1)); Y['d매출']=np.log(Y.점포당매출)-np.log(g.점포당매출.shift(1))
d=Y.dropna(subset=['d점포','d매출']); d=d[(g.점포.shift(1).reindex(d.index)>=3)&np.isfinite(d.d점포)&np.isfinite(d.d매출)]
mm=smf.wls('d매출~d점포+C(연)',d,weights=d.점포).fit(cov_type='cluster',cov_kwds={'groups':d.상권코드.astype('category').cat.codes})
print(f'  변화(상권 고정효과 = 전년 대비 변화, 2022–25, 상권-연 {len(d)}개): 치킨집 1% 증가 → 점포당 매출 {mm.params["d점포"]:+.2f}% (p={mm.pvalues["d점포"]:.3g})')
p=pd.read_csv(f'{P}/dong_chicken_panel.csv',dtype={'동':str}).dropna(subset=['d매출','d점포'])
mm2=smf.wls('d매출~d점포+C(연)',p,weights=p.점포).fit(cov_type='cluster',cov_kwds={'groups':p.동.astype('category').cat.codes})
print(f'  변화(행정동, dong_market.py 패널 {len(p)}개): {mm2.params["d점포"]:+.2f}% (p={mm2.pvalues["d점포"]:.3g}), 단순 상관 {pearsonr(p.d점포,p.d매출).statistic:+.2f}')
# 반 나눠 재현
for nm,q in [('2022–23',d[d.연<=2023]),('2024–25',d[d.연>=2024])]:
    k=smf.wls('d매출~d점포+C(연)',q,weights=q.점포).fit(); print(f'   {nm}: {k.params["d점포"]:+.2f}% (p={k.pvalues["d점포"]:.3g}, n={len(q)})')

print('\n[② 브랜드 생존×매출 산점도]')
B=pd.read_csv(f'{P}/brand_risk_return.csv'); print(f'  브랜드 {len(B)}개, 폐업률 vs 평균매출 순위상관 {spearmanr(B.폐업,B.매출_억).statistic:+.2f} (p={spearmanr(B.폐업,B.매출_억).pvalue:.2f})')
print(f'  95% 구간 폭 중앙값 {(B.hi-B.lo).median():.1f}%p, 가장 넓은 브랜드: '+', '.join(f'{r.브랜드} {r.lo:.0f}–{r.hi:.0f}%' for _,r in B.assign(w=B.hi-B.lo).sort_values('w',ascending=False).head(3).iterrows()))
ft=pd.read_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv'); bt=pd.read_csv(f'{P}/chicken_brand_table.csv')[['브랜드','ftc']].dropna()
ft=ft[ft.yr.between(2022,2024)&ft.brandNm.isin(bt.ftc)].groupby('brandNm').apply(lambda x:((x.ctrtEndCnt.fillna(0)+x.ctrtCncltnCnt.fillna(0)).sum()/x.frcsCnt.sum())).rename('이탈률')
Bx=B.merge(bt,on='브랜드').join(ft,on='ftc').dropna(subset=['이탈률'])
r=spearmanr(Bx.폐업,Bx.이탈률); print(f'  서울 3년 폐업(인허가) vs 공정위 전국 연 이탈률 2022–24: 순위상관 {r.statistic:+.2f} (p={r.pvalue:.3f}, 브랜드 {len(Bx)}개). 어긋나는 브랜드: '+', '.join(f'{r_.브랜드}(서울 {r_.폐업:.1f}%, 전국 {r_.이탈률*100:.1f}%)' for _,r_ in Bx.assign(dr=(Bx.폐업.rank()-Bx.이탈률.rank()).abs()).sort_values('dr',ascending=False).head(3).iterrows()))
Q=pd.read_csv(f'{P}/quadrant_brand.csv'); print(f'  사분면 두 기간 유지 {(Q.사분면_앞==Q.사분면_뒤).mean():.2f} (브랜드 {len(Q)}개), 매출 지속성 {spearmanr(Q.매출_앞,Q.매출_뒤).statistic:+.2f}, 폐업 지속성 {spearmanr(Q.폐업_앞,Q.폐업_뒤).statistic:+.2f}')

print('\n[③ 대조 검증: 상권 점포당 매출 상위 25% vs 나머지]')
def contrast(T,yr_sales,END,label,qs=(.25,)):
    S=Y[Y.연.isin(yr_sales)].groupby('상권코드').apply(lambda x:x.매출.sum()/x.점포분기.sum()).rename('ps')
    a=c[(c.open<=T)&(c.cc>T)&c.상권코드.notna()].join(S,on='상권코드').dropna(subset=['ps']).copy()
    a['y']=((a.cc<=END)&~a.재등록).astype(int); a['프랜']=a.프랜차이즈.astype(int); a['연차']=np.log1p((T-a.open).dt.days/365.25)
    a['pct']=a.상권코드.map(S.rank(pct=True))   # 상권 단위 백분위(점포 수 가중 아님)
    out=[]
    for q in qs:
        a['top']=(a.pct>=1-q).astype(int); mo=smf.logit('y~top+프랜+연차+면적',a).fit(disp=0)
        out.append((q,a[a.top==1].y.mean(),a[a.top==0].y.mean(),int(a.top.sum()),np.exp(mo.params['top']),mo.pvalues['top']))
    print(f'  {label}: 영업 중 {len(a)}곳, 기간 폐업률 {a.y.mean():.3f}')
    for q,p1,p0,n,orr,pv in out: print(f'    상위 {int(q*100)}%: {p1:.3f}({n}) vs {p0:.3f}, OR {orr:.2f} p={pv:.3g}')
    return a
a1=contrast(pd.Timestamp('2023-12-31'),[2021,2022,2023],pd.Timestamp('2026-11-04'),'시점 A: 2023-12-31, 매출 2021–23 → 폐업 2024-01~2026-11',(.2,.25,.3,.5))
a2=contrast(pd.Timestamp('2021-12-31'),[2021],pd.Timestamp('2024-12-31'),'시점 B: 2021-12-31, 매출 2021만 → 폐업 2022-01~2024-12',(.2,.25,.3,.5))
for nm,a in [('A',a1),('B',a2)]:
    for k,g in [('개인',a[a.프랜==0]),('프랜차이즈',a[a.프랜==1])]:
        top=g.pct>=.75; print(f'    시점 {nm} {k}: 상위 25% {g[top].y.mean():.3f}({int(top.sum())}) vs {g[~top].y.mean():.3f}')
rng=np.random.default_rng(0)
for nm,a in [('A',a1),('B',a2)]:
    codes=a.상권코드.unique(); obs=a[a.pct>=.75].y.mean()-a[a.pct<.75].y.mean(); null=[]
    pc=a.drop_duplicates('상권코드').set_index('상권코드').pct
    for _ in range(1000):
        sh=pd.Series(rng.permutation(pc.values),index=pc.index); tp=a.상권코드.map(sh)>=.75; null.append(a[tp].y.mean()-a[~tp].y.mean())
    print(f'  시점 {nm} 상권 매출 순위를 섞은 1,000회와 비교: 관측 차이 {obs*100:+.1f}%p, 섞은 차이 95% 범위 {np.percentile(null,2.5)*100:+.1f}~{np.percentile(null,97.5)*100:+.1f}%p, p={(np.sum(np.abs(null)>=abs(obs))+1)/1001:.3f}')

# ①-보강: 나눗셈 착시 확인. 점포당 매출 = 매출/점포 수라 점포 수 집계 오차만으로도 기울기가 음수가 될 수 있음
#  → x를 인허가 치킨집 수(다른 출처)로 바꾸고, y는 동의 치킨 매출 '총액' 변화. 기울기 < 1이면 점포가 늘어난 만큼 매출이 안 늘어 나눠 먹는 것
import glob
dm=load('dong_market'); cd=dm.c
yrs=range(2021,2026); rows=[]
for yr in yrs:
    t=pd.Timestamp(f'{yr}-07-01'); al_=cd[(cd.open<=t)&(cd.close.fillna(FAR)>t)]; rows.append(al_.groupby('동').size().rename(yr))
LIC=pd.concat(rows,axis=1).stack().rename('인허가점포').reset_index().rename(columns={'level_1':'연'})
SL=dm.sl.groupby(['동','연']).당월_매출_금액.sum().rename('총매출').reset_index()
z=LIC.merge(SL,on=['동','연']).sort_values(['동','연']); gz=z.groupby('동')
z['dx']=np.log(z.인허가점포)-np.log(gz.인허가점포.shift(1)); z['dy']=np.log(z.총매출)-np.log(gz.총매출.shift(1)); z['w']=gz.인허가점포.shift(1)
z=z[(z.w>=3)&np.isfinite(z.dx)&np.isfinite(z.dy)].dropna(subset=['dx','dy'])
k=smf.wls('dy~dx+C(연)',z,weights=z.w).fit(cov_type='cluster',cov_kwds={'groups':z.동.astype('category').cat.codes})
lo,hi=k.conf_int().loc['dx']
print(f'\n  ①-보강(행정동, x=인허가 치킨집 수 변화, y=치킨 매출 총액 변화, 동-연 {len(z)}개): 점포 1% 증가 → 총매출 {k.params["dx"]:+.2f}% [95% {lo:+.2f}~{hi:+.2f}]. 1보다 작으면 나눠 먹기')
