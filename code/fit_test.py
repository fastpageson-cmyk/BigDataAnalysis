# 브랜드 × 상권유형 궁합 검정: 3년 실제폐점, 2012~2023-11 개업, 상권 안 점포
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy import stats
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); H=3
TN={0:'시장·장년생활권',1:'오피스',2:'가족·주거여가',3:'2030핫플',4:'야간번화가'}
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl'); P=pd.read_pickle(f'{O}/data/processed/area_profile.pkl')
s=s.join(P,on='상권코드'); s['유형']=s.유형번호.map(TN)
e=s[(s.open.dt.year>=2012)&(s.open<=END-pd.DateOffset(years=H))&s.유형.notna()].copy()
e['y']=((e.close<=e.open+pd.DateOffset(years=H))&e.진짜폐점).astype(int)
print('n',len(e),'사건',e.y.sum(),'상권 수',e.상권코드.nunique())
print(e.pivot_table(index='업종',columns='유형',values='y',aggfunc='mean').round(3).to_string())
print(e.pivot_table(index='업종',columns='유형',values='y',aggfunc='size').to_string())
big=e.groupby('브랜드').y.agg(['size','sum']); keep=big[(big['size']>=150)].index; f=e[e.브랜드.isin(keep)].copy()
m0=smf.logit('y~C(브랜드)+C(유형)+C(연)',f).fit(disp=0,maxiter=300); m1=smf.logit('y~C(브랜드)*C(유형)+C(연)',f).fit(disp=0,maxiter=300,method='bfgs')
lr=2*(m1.llf-m0.llf); dfree=m1.df_model-m0.df_model; print(f'\n브랜드×유형 상호작용 LR검정 (브랜드 {len(keep)}개, n={len(f)}): χ²={lr:.1f}, df={dfree:.0f}, p={stats.chi2.sf(lr,dfree):.4f}')
g0=smf.logit('y~C(업종)+C(유형)+C(연)',e).fit(disp=0); g1=smf.logit('y~C(업종)*C(유형)+C(연)',e).fit(disp=0,method='bfgs',maxiter=500)
lr=2*(g1.llf-g0.llf); dfree=g1.df_model-g0.df_model; print(f'업종×유형 상호작용 LR검정: χ²={lr:.1f}, df={dfree:.0f}, p={stats.chi2.sf(lr,dfree):.4f}')
# 업종 매출 비중(수요) 효과: 같은 업종 매출 비중 높은 상권일수록 그 업종 폐점↓?
cat=['커피','치킨','버거·피자','베이커리·디저트','분식·한식','주점']
e['내업종매출비중']=[r[f'{r.업종}매출비중'] for r in e.itertuples()] if False else e.apply(lambda r: r[f'{r.업종}매출비중'],axis=1)
e['z']=e.groupby('업종').내업종매출비중.transform(lambda v:(v-v.mean())/v.std())
m=smf.logit('y~z+C(브랜드)+C(유형)+C(연)',e).fit(disp=0,method='bfgs',maxiter=500); print(f'내 업종 매출 비중 1SD: OR={np.exp(m.params.z):.2f}, p={m.pvalues.z:.3f}')
f[['브랜드','업종','유형','y']].to_pickle(f'{O}/data/processed/_fit_e.pkl')
# 브랜드별 유형 폐점률(표본 큰 브랜드)
t=f.pivot_table(index='브랜드',columns='유형',values='y',aggfunc='mean').round(3); n=f.pivot_table(index='브랜드',columns='유형',values='y',aggfunc='size')
print(t.to_string()); print(n.to_string())
