# 업종별 입지 민감도 비교: '동네의 과거 폐업률'이 새 가게의 3년 폐업을 얼마나 가르는가를 치킨과 다른 외식 업종에서 같은 방법으로 잰다.
#  y = 3년 안 인허가 폐업 − 점주 교체(같은 업종이 같은 자리 30m에 폐업 −60일~+1년 안 재등록). chicken_wide.py와 같은 규칙을 모든 업종에 적용.
#  동네 위험 = 500m 칸에서 2010–14 개업한 전 업종 음식점의 3년 실제 폐업률(평가 표본 개업 전인 2017년 말까지 결과 확정, 20곳 가중 수축).
#  평가 = 2018–22 개업. 통제 = 브랜드(30곳 이상 브랜드는 개별, 나머지 프랜차이즈 묶음, 개인) + 면적 + 개업연도.
# 출력: data/processed/industry_location.csv
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict,StratifiedKFold
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
a=pd.read_pickle(f'{P}/_lic_all.pkl').copy()
a['open']=pd.to_datetime(a['인허가일자'].astype(str).str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].astype(str).str.strip(),errors='coerce')
a['nm']=a['사업장명'].fillna('').str.replace(r'\s+','',regex=True)
a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce')
n0=len(a); a=a[a.X.notna()&a.Y.notna()&a.open.notna()].reset_index(drop=True); print(f'[정제] 좌표·인허가일 없는 행 제외 {n0-len(a)} / {n0}')
# 브랜드 사전: wide.py(업종별) + chicken_wide.py 추가분
src=open(f'{O}/code/wide.py',encoding='utf-8').read(); BD=eval(src[src.index('B={')+2:src.index('\nR=[]')])
BD['치킨'].update({'깐부':'깐부치킨','바른치킨':'바른치킨','땅땅':'땅땅치킨','둘둘':'둘둘치킨','가마로강정':'가마로','훌랄라':'훌랄라','부어치킨':'부어치킨',
          '오븐에빠진닭':'오븐에빠진닭','KFC':'(?i)KFC|케이에프씨','치킨플러스':'치킨플러스','꾸브라꼬':'꾸브라꼬','맥시칸':'맥시칸','썬더치킨':'썬더치킨'})
a['브랜드']=None; a['치킨브랜드']=False
for g,d in BD.items():
    for b,p in d.items():
        m=a.nm.str.contains(p,regex=True)&a.브랜드.isna(); a.loc[m,'브랜드']=b
        if g=='치킨': a.loc[m,'치킨브랜드']=True
KW='치킨|통닭|닭강정|후라이드|프라이드|양념반|순살'
NOT='코리안BBQ|바베큐|비비큐보쌈|맘스터치|닭갈비|삼계|닭한마리|찜닭|닭발|닭곰탕|닭칼국수|닭꼬치|백숙|닭도리'
ck=(a.nm.str.contains(KW,regex=True)|a.치킨브랜드|(a.업태구분명=='통닭(치킨)'))&~(a.nm.str.contains(NOT,regex=True)&~a.치킨브랜드)   # chicken_wide.py B_넓음
U=a.업태구분명
a['업종']=np.select([ck,U=='한식',U=='분식',U=='경양식',U.isin(['커피숍','까페']),U=='일식',U=='중국식',U.isin(['호프/통닭','정종/대포집/소주방']),U=='패스트푸드',U=='편의점'],
                  ['치킨','한식','분식','경양식','커피','일식','중식','주점','패스트푸드','편의점'],default='기타')
# 점주 교체 보정(업종 안에서)
a['재등록']=False
for g,s in a.groupby('업종'):
    T=cKDTree(s[['X','Y']].values); cl=s[s.close.notna()]; o=s.open.values; idx=s.index.values
    nb=T.query_ball_point(cl[['X','Y']].values,30); out=np.zeros(len(cl),bool)
    for i,(q,c0,me) in enumerate(zip(nb,cl.close.values,cl.index.values)):
        q=[j for j in q if idx[j]!=me]
        if q: oo=o[q]; out[i]=((oo>=c0-np.timedelta64(60,'D'))&(oo<=c0+np.timedelta64(365,'D'))).any()
    a.loc[cl.index,'재등록']=out
a['연']=a.open.dt.year
a['y신고']=(a.close<=a.open+pd.DateOffset(years=3)).astype(int); a['y']=(a.y신고.astype(bool)&~a.재등록).astype(int)
a['면적']=np.log1p(pd.to_numeric(a['소재지면적'],errors='coerce')); a['면적']=a.면적.fillna(a.groupby('업종').면적.transform('median'))
a['g']=(a.X//500).astype(int).astype(str)+'_'+(a.Y//500).astype(int).astype(str)
tr=a[a.연.between(2010,2014)]; m0=tr.y.mean(); cell=tr.groupby('g').y.agg(['sum','count'])
a['동네위험']=a.g.map((cell['sum']+20*m0)/(cell['count']+20)).fillna(m0); a['동네표본']=a.g.map(cell['count']).fillna(0)
te=a[a.연.between(2018,2022)].copy(); te['z']=(te.동네위험-te.동네위험.mean())/te.동네위험.std()
print(f'학습(동네 위험 계산) 2010–14 개업 {len(tr)}곳, 실제 폐업률 {m0:.3f}, 칸 {len(cell)}개 / 평가 2018–22 개업 {len(te)}곳')
print(f'평가 표본 중 동네 이력 20곳 미만 칸 비율 {(te.동네표본<20).mean():.1%}')

def boot(y,r,n=500):
    rng=np.random.default_rng(0); v=[]
    for _ in range(n):
        i=rng.integers(0,len(y),len(y)); v.append(roc_auc_score(y[i],r[i]))
    return np.percentile(v,[2.5,97.5])
def one(s,label):
    s=s.copy(); vc=s.브랜드.value_counts(); s['bg']=np.where(s.브랜드.isna(),'개인',np.where(s.브랜드.map(vc).fillna(0)>=30,s.브랜드,'기타프랜차이즈'))
    lo,hi=boot(s.y.values,s.동네위험.values); q=s.groupby(pd.qcut(s.동네위험,5,labels=False,duplicates='drop')).y.mean()
    r=dict(업종=label,n=len(s),신고폐업률=s.y신고.mean(),실제폐업률=s.y.mean(),점주교체비=1-s.y.sum()/max(s.y신고.sum(),1),프랜차이즈비=s.브랜드.notna().mean(),
           AUC_동네=roc_auc_score(s.y,s.동네위험),lo=lo,hi=hi,최저5분위=q.iloc[0],최고5분위=q.iloc[-1])
    try:
        f='y~z+면적+C(연)'+('+C(bg)' if s.bg.nunique()>1 else ''); m=smf.logit(f,s).fit(disp=0,maxiter=200)
        r.update(OR_SD_통제=np.exp(m.params['z']),p_통제=m.pvalues['z'])
    except Exception as e: r.update(OR_SD_통제=np.nan,p_통제=np.nan)
    Xb=pd.get_dummies(s[['bg','연']].astype(str),dtype=float).join(s[['면적']]); cv=StratifiedKFold(5,shuffle=True,random_state=0); L=LogisticRegression(C=1.0,max_iter=2000)
    pb=cross_val_predict(L,Xb,s.y,cv=cv,method='predict_proba')[:,1]; pl=cross_val_predict(L,Xb.join(s[['z']]),s.y,cv=cv,method='predict_proba')[:,1]
    rng=np.random.default_rng(1); y=s.y.values; d=[]
    for _ in range(500):
        i=rng.integers(0,len(y),len(y)); d.append(roc_auc_score(y[i],pl[i])-roc_auc_score(y[i],pb[i]))
    r.update(AUC_브랜드면적=roc_auc_score(y,pb),AUC_더하기동네=roc_auc_score(y,pl),증가=roc_auc_score(y,pl)-roc_auc_score(y,pb),증가lo=np.percentile(d,2.5),증가hi=np.percentile(d,97.5))
    return r
R=[one(te[te.업종==k],k) for k in ['치킨','한식','분식','경양식','커피','일식','중식','주점','패스트푸드','편의점']]
c=te[te.업종=='치킨']; ar=np.expm1(c.면적)
R+=[one(c[c.브랜드.notna()],'치킨·프랜차이즈'),one(c[c.브랜드.isna()],'치킨·개인'),one(c[ar<33],'치킨·33㎡ 미만'),one(c[(ar>=33)&(ar<66)],'치킨·33–66㎡'),one(c[ar>=66],'치킨·66㎡ 이상')]
R=pd.DataFrame(R); pd.set_option('display.width',300); pd.set_option('display.max_columns',40)
print(R.round(3).to_string(index=False)); R.round(4).to_csv(f'{P}/industry_location.csv',index=False,encoding='utf-8-sig')

# 민감도: 점주 교체 보정 없이(신고 폐업 그대로) 같은 비교. 한식처럼 밀집한 업종은 30m 재등록 규칙이 과하게 걸릴 수 있어 확인.
print('\n[민감도] 신고 폐업 기준(보정 없음): 동네 위험도 신고 폐업으로 다시 계산')
c2=tr.groupby('g').y신고.agg(['sum','count']); m2=tr.y신고.mean(); te['동네위험2']=te.g.map((c2['sum']+20*m2)/(c2['count']+20)).fillna(m2); S=[]
for k in ['치킨','한식','분식','경양식','커피','일식','중식','주점','패스트푸드','편의점']:
    s=te[te.업종==k]; lo,hi=boot(s.y신고.values,s.동네위험2.values); q=s.groupby(pd.qcut(s.동네위험2,5,labels=False,duplicates='drop')).y신고.mean()
    S.append(dict(업종=k,n=len(s),AUC_신고=roc_auc_score(s.y신고,s.동네위험2),lo=lo,hi=hi,최저5분위=q.iloc[0],최고5분위=q.iloc[-1],
                  AUC_실제y_신고위험=roc_auc_score(s.y,s.동네위험2)))
S=pd.DataFrame(S); print(S.round(3).to_string(index=False)); S.round(4).to_csv(f'{P}/industry_location_sens.csv',index=False,encoding='utf-8-sig')
