# brand_fit 표본(2010–21 개업 프랜차이즈)에서 폐업 기준 3년 vs 5년: 폐업 수, 사람유형 효과, 브랜드×입지 궁합 p 비교
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf; from scipy.stats import chi2
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); spec=importlib.util.spec_from_file_location('bf',f'{O}/code/brand_fit.py'); bf=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bf)
f=bf.f.copy(); f['y3']=((f.close<=f.open+pd.DateOffset(years=3))&~f.재등록).astype(int); f['y5']=f.y
print('n',len(f),'폐업수 3년',f.y3.sum(),'5년',f.y5.sum(), '비율',round(f.y3.mean(),3),round(f.y5.mean(),3))
print('5년 폐업 중 3~5년 사이 비중',round((f.y5.sum()-f.y3.sum())/f.y5.sum(),3))
for y in ['y3','y5']:
    m0=smf.logit(f'{y}~C(브랜드)+면적+C(연)',f).fit(disp=0,method='bfgs',maxiter=2000); m1=smf.logit(f'{y}~C(브랜드)+면적+C(연)+C(사람유형)',f).fit(disp=0,method='bfgs',maxiter=2000)
    s=2*(m1.llf-m0.llf); print(y,'사람유형 효과 p',round(chi2.sf(s,len(m1.params)-len(m0.params)),4),f.groupby('사람유형')[y].mean().round(3).to_dict())
    for v in ['공식유형','사람유형','면적구간']:
        b0=smf.logit(f'{y}~C(브랜드)+C({v})+C(연)',f).fit(disp=0,method='bfgs',maxiter=3000); b1=smf.logit(f'{y}~C(브랜드)*C({v})+C(연)',f).fit(disp=0,method='bfgs',maxiter=3000)
        s=2*(b1.llf-b0.llf); print('  궁합',v,'p',round(chi2.sf(s,len(b1.params)-len(b0.params)),3))
t=f.groupby('브랜드')[['y3','y5']].agg(['sum','size']); print(t[[('y3','sum'),('y5','sum'),('y5','size')]].to_string())
