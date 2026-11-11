# 궁합 검정 민감도: 기간(3·5년)·폐업 정의(실제/인허가 전체)·생존시간(Cox) 바꿔도 브랜드×상권유형 상호작용이 있나
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from statsmodels.duration.hazard_regression import PHReg
from scipy import stats
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
TN={0:'시장·장년생활권',1:'오피스',2:'가족·주거여가',3:'2030핫플',4:'야간번화가'}
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/area_profile.pkl'),on='상권코드'); s['유형']=s.유형번호.map(TN)
s=s[s.유형.notna()&(s.open.dt.year>=2010)]
def lr(f,a,b):
    m0=smf.logit(a,f).fit(disp=0,method='bfgs',maxiter=800); m1=smf.logit(b,f).fit(disp=0,method='bfgs',maxiter=800)
    x=2*(m1.llf-m0.llf); d=m1.df_model-m0.df_model; return f'χ²={x:.1f} df={d:.0f} p={stats.chi2.sf(x,d):.3f}'
for H in [3,5]:
    for yc in ['진짜폐점',None]:
        e=s[s.open<=END-pd.DateOffset(years=H)].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=H))&(e[yc] if yc else True)).astype(int)
        keep=e.브랜드.value_counts(); f=e[e.브랜드.isin(keep[keep>=150].index)]
        print(f'H={H} {yc or "인허가폐업"} n={len(f)} 사건={f.y.sum()} 브랜드={f.브랜드.nunique()} | 브랜드×유형 {lr(f,"y~C(브랜드)+C(유형)+C(연)","y~C(브랜드)*C(유형)+C(연)")} | 업종×유형 {lr(e,"y~C(브랜드)+C(유형)+C(연)","y~C(브랜드)+C(업종):C(유형)+C(유형)+C(연)")}')
# Cox: 전체 추적기간 생존시간
e=s.copy(); ev=e.진짜폐점&e.close.notna(); t=((e.close.where(ev)).fillna(END)-e.open).dt.days.clip(lower=1)
keep=e.브랜드.value_counts(); m=e.브랜드.isin(keep[keep>=150].index); e=e[m]; ev=ev[m]; t=t[m]
X0=pd.get_dummies(e[['브랜드','유형']],drop_first=True,dtype=float)
I=pd.get_dummies(e.브랜드+'×'+e.유형,dtype=float); I=I.loc[:,I.sum()>=10]
r0=PHReg(t.values,X0.values,status=ev.values.astype(int)).fit(); X1=pd.concat([X0,I],axis=1); X1=X1.loc[:,X1.std()>0]
from numpy.linalg import matrix_rank
X1=X1.iloc[:,:matrix_rank(X1.values)] if matrix_rank(X1.values)<X1.shape[1] else X1
r1=PHReg(t.values,X1.values,status=ev.values.astype(int)).fit()
x=2*(r1.llf-r0.llf); d=X1.shape[1]-X0.shape[1]; print(f'Cox 전체기간 n={len(e)} 사건={ev.sum()} 브랜드×유형 χ²={x:.1f} df={d} p={stats.chi2.sf(x,d):.3f}')
