# 이동 데이터가 치킨집 3년 실제 폐업을 더 설명하는가? (chicken_wide.py B_넓음 표본, 2018–22 개업)
# 변수는 모두 개업 전년도 값(미래 정보 누수 없음): 행정동 생활인구(낮·저녁·밤), 저녁인구/동 치킨집 수, 반경 500m 역 저녁 하차
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
spec=importlib.util.spec_from_file_location('cw',f'{O}/code/chicken_wide.py'); cw=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(cw)
d=cw.D['B_넓음'].copy()

# 행정동 배정: 상가정보 최근접 5곳 다수결(100m 안에 없으면 결측)
b=pd.read_csv(f'{P}/sbiz_dong_points.csv',dtype={'동':str}); T=cKDTree(b[['X','Y']].values)
def dong(X,Y):
    dist,idx=T.query(np.c_[X,Y],k=5); lab=b.동.values[idx]
    out=pd.Series([pd.Series(r).mode().iat[0] for r in lab]); out[dist[:,0]>100]=None; return out.values
d['동']=dong(d.X.values,d.Y.values)
# 동별·시점별 영업 중 치킨집 수(같은 B 정의 전체 점포 기준)
c=cw.a[cw.defs['B_넓음']].copy(); c['X']=pd.to_numeric(c['좌표정보(X)'],errors='coerce'); c['Y']=pd.to_numeric(c['좌표정보(Y)'],errors='coerce'); c=c.dropna(subset=['X'])
c['동']=dong(c.X.values,c.Y.values)
def active_in_dong(r):
    m=(c.동.values==r.동)&(c.open.values<np.datetime64(r.open))&(c.close.fillna(pd.Timestamp('2100-01-01')).values>np.datetime64(r.open)); return m.sum()
d=d[d.연.between(2018,2022)&d.동.notna()].copy()
d['동치킨']=[active_in_dong(r) for r in d.itertuples()]

lp=pd.read_csv(f'{P}/living_pop_dong_year.csv',dtype={'동':str}); lp['연']=lp.연+1   # 개업 전년도 값
d=d.merge(lp,on=['동','연'],how='left')
sm=pd.read_csv(f'{P}/subway_station_year.csv'); sm['연']=sm.연+1
ev=[];near=[]
for r in d.itertuples():
    q=sm[sm.연==r.연]; dist=np.hypot(q.X.values-r.X,q.Y.values-r.Y); ev.append(q.저녁하차.values[dist<=500].sum()); near.append(dist.min())
d['역저녁하차500']=ev; d['역거리']=near
d['l저녁']=np.log(d.저녁); d['저녁낮비']=d.저녁/d.낮; d['l저녁per치킨']=np.log(d.저녁/(d.동치킨+1)); d['l역저녁']=np.log1p(d.역저녁하차500); d['l밀도']=np.log1p(d.밀도)
d=d.dropna(subset=['저녁']); print(f'표본 {len(d)}곳(2018–22 개업, 동 매칭), 프랜차이즈 {int(d.프랜차이즈.sum())}, 3년 실제 폐업률 {d.y_실제.mean():.3f}')
d.drop(columns=[x for x in d.columns if x in('지번주소','사업장명','nm')]).to_csv(f'{P}/chicken_mobility_sample.csv',index=False,encoding='utf-8-sig')

base='y_실제~프랜차이즈+면적+l밀도+C(연)'; MOB='+l저녁+저녁낮비+l저녁per치킨+l역저녁'
def fit(f,data): return smf.logit(f,data).fit(disp=0)
m0=fit(base,d); m1=fit(base+MOB,d)
lr=2*(m1.llf-m0.llf); print(f'\n[전체] 이동 변수 4개 묶음 우도비검정 χ²={lr:.2f}, p={chi2.sf(lr,4):.3f}')
print(pd.DataFrame({'OR':np.exp(m1.params),'p':m1.pvalues}).loc[['l저녁','저녁낮비','l저녁per치킨','l역저녁','l밀도','프랜차이즈[T.True]','면적']].round(3).to_string())
# 시간 분리 예측: 2018–20 학습 → 2021–22 평가
tr=d.연<=2020; te=~tr
for nm,f in [('기본(프랜·면적·밀도)',base.replace('+C(연)','')),('+이동 4개',base.replace('+C(연)','')+MOB)]:
    m=fit(f,d[tr]); print(f'평가 AUC {nm}: {roc_auc_score(d[te].y_실제,m.predict(d[te])):.3f}')
# 프랜차이즈 안에서(브랜드 고정효과): 50대 가맹 창업 상황
f=d[d.프랜차이즈].copy(); top=f.브랜드.value_counts(); f=f[f.브랜드.isin(top[top>=20].index)]
b0=fit('y_실제~C(브랜드)+면적+l밀도+C(연)',f); b1=fit('y_실제~C(브랜드)+面적'.replace('面','면')+'+l밀도+C(연)'+MOB,f)
lr=2*(b1.llf-b0.llf); print(f'\n[프랜차이즈 {len(f)}곳, 브랜드 고정효과] 이동 묶음 χ²={lr:.2f}, p={chi2.sf(lr,4):.3f}')
print(pd.DataFrame({'OR':np.exp(b1.params),'p':b1.pvalues}).loc[['l저녁','저녁낮비','l저녁per치킨','l역저녁']].round(3).to_string())
# 해석용 구간표: 저녁인구/치킨집 5분위별 폐업률
d['수요구간']=pd.qcut(d.l저녁per치킨,5,labels=['최저','낮음','중간','높음','최고'])
print('\n저녁 생활인구 ÷ 동 치킨집 수, 5분위별 3년 실제 폐업률'); print(d.groupby(['수요구간','프랜차이즈'],observed=True).y_실제.agg(['mean','size']).unstack().round(3).to_string())
