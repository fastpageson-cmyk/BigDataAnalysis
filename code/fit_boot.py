# 궁합 LR검정의 모수적 부트스트랩: 상호작용 없는 모형에서 y를 다시 뽑아 LR 분포를 만들고 관측값과 비교(희소 셀 과대평가 보정)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
TN={0:'시장·장년생활권',1:'오피스',2:'가족·주거여가',3:'2030핫플',4:'야간번화가'}
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/area_profile.pkl'),on='상권코드'); s['유형']=s.유형번호.map(TN)
s=s[s.유형.notna()&(s.open.dt.year>=2010)]
A,B='y~C(브랜드)+C(유형)+C(연)','y~C(브랜드)*C(유형)+C(연)'
def LR(f):
    m0=smf.logit(A,f).fit(disp=0,method='bfgs',maxiter=800); m1=smf.logit(B,f).fit(disp=0,method='bfgs',maxiter=800); return 2*(m1.llf-m0.llf),m0
for H,yc in [(5,'진짜폐점'),(3,None)]:
    e=s[s.open<=END-pd.DateOffset(years=H)].copy(); e['y']=((e.close<=e.open+pd.DateOffset(years=H))&(e[yc] if yc else True)).astype(int)
    k=e.브랜드.value_counts(); f=e[e.브랜드.isin(k[k>=150].index)].copy()
    obs,m0=LR(f); p0=m0.predict(f).values; rs=np.random.RandomState(0); sim=[]
    for _ in range(200):
        g=f.copy(); g['y']=(rs.rand(len(g))<p0).astype(int); sim.append(LR(g)[0])
    sim=np.array(sim); print(f'H={H} {yc or "인허가폐업"}: 관측 LR={obs:.1f}, 귀무 분포 중앙 {np.median(sim):.1f} 95% {np.percentile(sim,95):.1f}, 부트스트랩 p={(np.sum(sim>=obs)+1)/(len(sim)+1):.3f}')
