# 브랜드 선행 신호: 전국 치킨 브랜드×연도 패널(공정위 가맹점 현황 15110241)에서 '올해 신호 → 다음 해 이탈률'을 본다.
#  이탈 = 계약종료 + 계약해지. 분모는 연초 가맹점 수(전년 말 가맹점 수). yr는 등록 연도(실적은 yr-1년).
#  정제: ① 브랜드명 표기 통일 ② 같은 브랜드·연도 중복 중 가맹점 많은 행 ③ 0으로 찍힌 결측 제거
#        ④ 항등식(전년 가맹점+신규-종료-해지=올해 가맹점) 오차 5% 초과 연결 제외 ⑤ 연초 30곳 미만 제외
# 출력: data/processed/ftc_chicken_panel.csv, brand_signal_tests.csv, brand_entry.csv(주요 브랜드 신규 개점 vs 명의변경)
import os,re,html,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
pd.set_option('display.width',250)
LOG=[]
def log(step,n,why): LOG.append((step,n,why)); print(f'[정제] {step}: {n} ({why})')

raw=pd.read_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv',low_memory=False)
c=raw[raw.indutyMlsfcNm=='치킨'].copy(); log('치킨 업종 행',len(c),'원자료 76,846행 중')
def norm(s):
    s=re.sub(r'\(.*?\)','',html.unescape(str(s))).replace(' ','').lower()
    return 'bbq' if re.search('비비큐|bbq',s) else s
c['b']=c.brandNm.map(norm)
c.loc[c.brandNm.str.contains('비비큐|BBQ|bbq',na=False)&~c.corpNm.str.contains('제너시스',na=False),'b']=c.brandNm  # 제너시스 외 '비비큐' 포함 상호는 합치지 않음
log('표기 통일로 합쳐진 브랜드명',c.brandNm.nunique()-c.b.nunique(),'괄호·공백·대소문자, BBQ 표기 5종')
z=(c[['frcsCnt','newFrcsRgsCnt','ctrtEndCnt','ctrtCncltnCnt']].sum(axis=1)==0); log('전 항목 0인 행 제거',int(z.sum()),'미제출·미운영, 결측으로 봄'); c=c[~z]
n0=len(c); c=c.sort_values('frcsCnt',ascending=False).drop_duplicates(['b','yr']); log('같은 브랜드·연도 중복 제거',n0-len(c),'가맹점 수 많은 행 유지')
c=c.sort_values(['b','yr']); g=c.groupby('b')
for k in ['frcsCnt','avrgSlsAmt','yr']: c['p_'+k]=g[k].shift(1)
c['이탈']=c.ctrtEndCnt+c.ctrtCncltnCnt
c['연결']=(c.p_yr==c.yr-1)&(c.p_frcsCnt>0)
c['오차']=(c.frcsCnt-(c.p_frcsCnt+c.newFrcsRgsCnt-c.이탈)).abs()/c.p_frcsCnt
k=c[c.연결&(c.p_frcsCnt>=30)]
log('연초 30곳 이상 연속 브랜드-연도',len(k),f'항등식 정확 일치 {(k.오차==0).mean():.1%}, 5% 이내 {(k.오차<=.05).mean():.1%}')
bad=k[(k.오차>.05)]; log('항등식 오차 5% 초과 제외',len(bad),f'그중 올해 가맹점 수 0 기재 {(bad.frcsCnt==0).sum()}건')
c['유효']=c.연결&(c.p_frcsCnt>=30)&(c.오차<=.05)
c['이탈률']=c.이탈/c.p_frcsCnt; c['신규율']=c.newFrcsRgsCnt/c.p_frcsCnt; c['명의변경률']=c.nmChgCnt/c.p_frcsCnt
c['매출']=c.avrgSlsAmt.replace(0,np.nan); c['매출변화']=np.log(c.매출/c.p_avrgSlsAmt.replace(0,np.nan))
c.loc[~c.유효,['이탈률','신규율','명의변경률','매출변화']]=np.nan
print('유효 행 이탈률 최대',round(c.이탈률.max(),3),'/ 1 초과',int((c.이탈률>1).sum()))
g=c.groupby('b')
c['다음_이탈']=g.이탈.shift(-1); c['다음_유효']=g.유효.shift(-1).fillna(False); c['다음_이탈률']=g.이탈률.shift(-1)
p=c[c.유효&c.다음_유효].copy(); p['n']=p.frcsCnt; p['l규모']=np.log(p.frcsCnt)
p=p[p.frcsCnt>=30]; log('분석 패널(올해·다음 해 모두 유효)',len(p),f'브랜드 {p.b.nunique()}개, 신호 연도 {p.yr.min()}–{p.yr.max()}')
c[['b','brandNm','corpNm','yr','frcsCnt','newFrcsRgsCnt','ctrtEndCnt','ctrtCncltnCnt','nmChgCnt','매출','유효','이탈률','신규율','명의변경률','매출변화','다음_이탈률']].to_csv(f'{P}/ftc_chicken_panel.csv',index=False,encoding='utf-8-sig')

print('\n[1] 4분위별 다음 해 이탈률 (브랜드 평균 / 가맹점 가중)')
def qt(col,lab):
    s=p.dropna(subset=[col]).copy(); s['q']=pd.qcut(s[col],4,labels=['Q1','Q2','Q3','Q4'])
    t=s.groupby('q').apply(lambda x:pd.Series({'브랜드연도':len(x),lab:x[col].mean(),'다음해이탈률_평균':x.다음_이탈률.mean(),'다음해이탈률_가중':x.다음_이탈.sum()/x.n.sum()}))
    print(t.round(3).to_string()); return t.assign(신호=lab)
T=pd.concat([qt('신규율','신호평균'),qt('매출변화','신호평균'),qt('명의변경률','신호평균'),qt('이탈률','신호평균')])

print('\n[2] 이항 회귀: 다음 해 이탈 수 / 올해 말 가맹점 수 (브랜드 군집 표준오차, 표준화 계수)')
q=p.dropna(subset=['매출변화']).copy(); X=['이탈률','신규율','명의변경률','매출변화','l규모']
for x in X: q['z_'+x]=(q[x]-q[x].mean())/q[x].std()
q['y1']=q.다음_이탈.clip(upper=q.n); q['y0']=q.n-q.y1
def fit(d,xs):
    return sm.GLM(d[['y1','y0']],sm.add_constant(pd.get_dummies(d.yr,prefix='yr',drop_first=True,dtype=float).join(d[xs])),family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':d.b})
m=fit(q,['z_'+x for x in X]); R=[]
for x in X: R.append(dict(표본='전체',n=len(q),지표=x,OR_per_SD=np.exp(m.params['z_'+x]),p=m.pvalues['z_'+x]))
q2=q[q.frcsCnt>=100]; m2=fit(q2,['z_'+x for x in X])
for x in X: R.append(dict(표본='가맹점100+',n=len(q2),지표=x,OR_per_SD=np.exp(m2.params['z_'+x]),p=m2.pvalues['z_'+x]))
mo=smf.ols('다음_이탈률~'+'+'.join('z_'+x for x in X)+'+C(yr)',q).fit(cov_type='cluster',cov_kwds={'groups':q.b})
for x in X: R.append(dict(표본='전체(가중 없음 OLS, 계수=%p)',n=len(q),지표=x,OR_per_SD=mo.params['z_'+x]*100,p=mo.pvalues['z_'+x]))
R=pd.DataFrame(R); print(R.round(4).to_string(index=False)); R.to_csv(f'{P}/brand_signal_tests.csv',index=False,encoding='utf-8-sig')

print('\n[3] 시간 분할: 신호 연도 ≤2021 학습 → 2022 이후 평가. 기준선 = 올해 이탈률 그대로')
tr=q[q.yr<=2021]; te=q[q.yr>=2022].copy()
def fit2(d,xs): return sm.GLM(d[['y1','y0']],sm.add_constant(d[xs]),family=sm.families.Binomial()).fit()
ma=fit2(tr,['이탈률']); mb=fit2(tr,['이탈률','신규율','매출변화','l규모'])
te['기준']=ma.predict(sm.add_constant(te[['이탈률']])); te['신호']=mb.predict(sm.add_constant(te[['이탈률','신규율','매출변화','l규모']]))
hi=(te.다음_이탈률>=te.다음_이탈률.quantile(.75)).astype(int)
rng=np.random.default_rng(0); B=[]
for _ in range(2000):
    i=rng.integers(0,len(te),len(te)); B.append(roc_auc_score(hi.values[i],te.신호.values[i])-roc_auc_score(hi.values[i],te.기준.values[i]))
print(f'학습 {len(tr)} / 평가 {len(te)} 브랜드-연도 (브랜드 {te.b.nunique()})')
for k in ['기준','신호']: print(f' {k}: 순위상관 {spearmanr(te[k],te.다음_이탈률)[0]:.3f}, MAE {np.abs(te[k]-te.다음_이탈률).mean():.4f}, 이탈 상위 25% 브랜드 맞히기 AUC {roc_auc_score(hi,te[k]):.3f}')
print(f' AUC 차 {np.mean(B):+.3f} (95% {np.percentile(B,2.5):+.3f}~{np.percentile(B,97.5):+.3f})')
te['q']=pd.qcut(te.신호,4,labels=['안전','보통','주의','위험']); print(te.groupby('q').apply(lambda x:pd.Series({'브랜드연도':len(x),'예측':x.신호.mean(),'실제_다음해이탈률':x.다음_이탈.sum()/x.n.sum()})).round(3).to_string())

print('\n[4] 주요 브랜드 최근 실적연도 신규 개점 vs 명의변경 (yr=2025 등록, 2024년 실적)')
K={'교촌치킨':'교촌','지코바양념치킨':'지코바','굽네치킨':'굽네','처갓집양념치킨':'처갓집','60계':'60계','푸라닭':'푸라닭','비에이치씨':'bhc','bbq':'BBQ','노랑통닭':'노랑통닭',
   '네네치킨':'네네','깐부치킨':'깐부','호식이두마리치킨':'호식이','자담치킨':'자담','멕시카나':'멕시카나'}
e=c[c.b.isin(K)&(c.yr==2025)].assign(브랜드=lambda d:d.b.map(K)).set_index('브랜드')[['frcsCnt','newFrcsRgsCnt','nmChgCnt','이탈','이탈률','신규율','명의변경률','매출변화']]
e['신규/명의변경']=e.newFrcsRgsCnt/e.nmChgCnt
s=pd.read_csv(f'{P}/chicken_brand_table.csv').set_index('브랜드').서울3년실제폐업
e=e.join(s).sort_values('서울3년실제폐업'); print(e.round(3).to_string()); e.round(4).to_csv(f'{P}/brand_entry.csv',encoding='utf-8-sig')
print(f"서울 3년 폐업률 ↔ 신규/명의변경 비 순위상관 {spearmanr(e.서울3년실제폐업,e['신규/명의변경'],nan_policy='omit')[0]:.2f} (p={spearmanr(e.서울3년실제폐업,e['신규/명의변경'],nan_policy='omit')[1]:.3f}, n={e['신규/명의변경'].notna().sum()})")
al=c[(c.yr==2025)&(c.frcsCnt>=100)]; print(f"가맹점 100곳 이상 {len(al)}개 브랜드: 신규 개점 합 {int(al.newFrcsRgsCnt.sum())}, 명의변경 합 {int(al.nmChgCnt.sum())}, 이탈 합 {int(al.이탈.sum())}")
pd.DataFrame(LOG,columns=['처리','건수','근거']).to_csv(f'{P}/brand_signal_cleanlog.csv',index=False,encoding='utf-8-sig')
