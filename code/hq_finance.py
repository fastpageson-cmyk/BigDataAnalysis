# 가맹본부 재무(공정위 15125456) → 치킨 브랜드 다음 해 이탈률. brand_signal.py의 정제 패널(ftc_chicken_panel.csv)에 붙인다.
#  연결: (등록연도, 브랜드명) → 브랜드 목록(15125467)의 가맹본부 번호 → 같은 등록연도 재무(회계연도 = 등록연도-1).
#  재무 값은 '하한~상한' 구간(천원)으로 공개된다 → 가운데 값을 쓴다. 재무제표 미작성(fnlttWrtYn=N)·0은 결측.
#  지표: 본부 영업이익률, 부채비율(부채/자본), 본부 매출÷가맹점 수(가맹점 1곳당 본부 매출), 자본잠식(자본<0).
# 출력: data/processed/hq_finance_tests.csv
import os,re,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
pd.set_option('display.width',250)
f=pd.read_csv(f'{O}/data/raw/ftc_disclosure/finance.csv',dtype=str)
def mid(s):
    if pd.isna(s): return np.nan
    v=[float(x) for x in re.findall(r'-?\d+',str(s))]
    return np.nan if not v else float(np.mean(v))
for k in ['assetsAmtScopeVal','caplAmtScopeVal','debtAmtScopeVal','slsAmtScopeVal','bsnProfitAmtScopeVal','thstrmNtpfAmtScopeVal']: f[k]=f[k].map(mid)
print('재무 행',len(f),'재무제표 작성 Y 비율',(f.fnlttWrtYn=='Y').mean().round(3))
f=f[(f.fnlttWrtYn=='Y')&(f.slsAmtScopeVal>0)].copy(); f['yr']=f.jngBizCrtraYr.astype(int)
f=f.sort_values('slsAmtScopeVal',ascending=False).drop_duplicates(['yr','jnghdqrtrsMnno'])
L=pd.read_csv(f'{O}/data/raw/ftc_brand_list_2017_2025.csv',dtype=str); L=L[L.indutyMlsfcNm=='치킨']; L['yr']=L.jngBizCrtraYr.astype(int)
L=L.drop_duplicates(['yr','brandNm'])[['yr','brandNm','jnghdqrtrsMnno']]
p=pd.read_csv(f'{P}/ftc_chicken_panel.csv')
p=p.merge(L,on=['yr','brandNm'],how='left').merge(f[['yr','jnghdqrtrsMnno','assetsAmtScopeVal','caplAmtScopeVal','debtAmtScopeVal','slsAmtScopeVal','bsnProfitAmtScopeVal']],on=['yr','jnghdqrtrsMnno'],how='left')
p['영업이익률']=p.bsnProfitAmtScopeVal/p.slsAmtScopeVal
p['부채비율']=p.debtAmtScopeVal/p.caplAmtScopeVal.where(p.caplAmtScopeVal>0)
p['자본잠식']=(p.caplAmtScopeVal<0).astype(float).where(p.caplAmtScopeVal.notna())
p['점당본부매출']=np.log(p.slsAmtScopeVal/p.frcsCnt.where(p.frcsCnt>0))
p['l부채비율']=np.log1p(p.부채비율.clip(lower=0))
q=p[p.유효&p.다음_이탈률.notna()&(p.frcsCnt>=30)].copy()
print(f'분석 패널 {len(q)} 브랜드-연도 중 본부 번호 연결 {q.jnghdqrtrsMnno.notna().mean():.1%}, 재무 연결 {q.slsAmtScopeVal.notna().mean():.1%}')
q=q[q.slsAmtScopeVal.notna()&q.영업이익률.between(-2,1)].copy()
q['n']=q.frcsCnt; q['다음_이탈']=(q.다음_이탈률*q.n).round(); q['y1']=q.다음_이탈.clip(upper=q.n); q['y0']=q.n-q.y1; q['l규모']=np.log(q.n)
print(f'재무 있는 패널 {len(q)} (브랜드 {q.b.nunique()}), 자본잠식 {int(q.자본잠식.sum())}건')
print('\n[1] 4분위별 다음 해 이탈률(가맹점 가중)')
for x in ['영업이익률','l부채비율','점당본부매출']:
    s=q.dropna(subset=[x]); s['g']=pd.qcut(s[x],4,labels=['Q1','Q2','Q3','Q4'])
    print(x, s.groupby('g').apply(lambda d:round(d.y1.sum()/d.n.sum(),3)).to_dict(), 'n',len(s))
s=q.dropna(subset=['자본잠식']); print('자본잠식', s.groupby('자본잠식').apply(lambda d:(len(d),round(d.y1.sum()/d.n.sum(),3))).to_dict())
print('\n[2] 이항 회귀(브랜드 군집 SE, 표준화): 올해 이탈률·규모 통제 위에 재무 지표')
R=[]
s=q.dropna(subset=['영업이익률','l부채비율','점당본부매출']).copy()
for x in ['이탈률','l규모','영업이익률','l부채비율','점당본부매출']: s['z_'+x]=(s[x]-s[x].mean())/s[x].std()
def fit(d,xs): return sm.GLM(d[['y1','y0']],sm.add_constant(pd.get_dummies(d.yr,prefix='yr',drop_first=True,dtype=float).join(d[xs])),family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.b})
for xs,lab in [(['z_영업이익률','z_l부채비율','z_점당본부매출'],'재무만'),(['z_이탈률','z_l규모','z_영업이익률','z_l부채비율','z_점당본부매출'],'올해 이탈률·규모 통제')]:
    m=fit(s,xs)
    for x in xs: R.append(dict(모형=lab,n=len(s),브랜드=s.b.nunique(),지표=x[2:],OR_per_SD=np.exp(m.params[x]),p=m.pvalues[x]))
R=pd.DataFrame(R); print(R.round(4).to_string(index=False)); R.to_csv(f'{P}/hq_finance_tests.csv',index=False,encoding='utf-8-sig')
print('\n[3] 시간 분할(≤2021 학습 → 2022+ 평가): 다음 해 이탈 상위 25% 브랜드 맞히기')
tr=s[s.yr<=2021]; te=s[s.yr>=2022].copy(); hi=(te.다음_이탈률>=te.다음_이탈률.quantile(.75)).astype(int)
def fit2(d,xs): return sm.GLM(d[['y1','y0']],sm.add_constant(d[xs]),family=sm.families.Binomial()).fit()
for xs,lab in [(['이탈률','l규모'],'기준(올해 이탈률+규모)'),(['영업이익률','l부채비율','점당본부매출'],'재무만'),(['이탈률','l규모','영업이익률','l부채비율','점당본부매출'],'기준+재무')]:
    te[lab]=fit2(tr,xs).predict(sm.add_constant(te[xs]))
rng=np.random.default_rng(0); B=[]
for _ in range(2000):
    i=rng.integers(0,len(te),len(te)); B.append(roc_auc_score(hi.values[i],te['기준+재무'].values[i])-roc_auc_score(hi.values[i],te['기준(올해 이탈률+규모)'].values[i]))
for lab in ['기준(올해 이탈률+규모)','재무만','기준+재무']: print(f' {lab}: AUC {roc_auc_score(hi,te[lab]):.3f}')
print(f' 재무 추가 효과 {np.mean(B):+.3f} (95% {np.percentile(B,2.5):+.3f}~{np.percentile(B,97.5):+.3f}), 평가 {len(te)} 브랜드-연도')
K={'교촌치킨':'교촌','지코바양념치킨':'지코바','굽네치킨':'굽네','처갓집양념치킨':'처갓집','60계':'60계','푸라닭':'푸라닭','비에이치씨':'bhc','bbq':'BBQ','노랑통닭':'노랑통닭','네네치킨':'네네','깐부치킨':'깐부','호식이두마리치킨':'호식이','자담치킨':'자담','멕시카나':'멕시카나'}
t=p[p.b.isin(K)&(p.yr==2025)].assign(브랜드=lambda d:d.b.map(K)).set_index('브랜드')[['영업이익률','부채비율','점당본부매출','이탈률']]
t['점당본부매출_백만']=np.exp(t.점당본부매출)/1e3; print('\n[4] 주요 브랜드 본부 2024 회계연도'); print(t.drop(columns='점당본부매출').round(3).to_string())
