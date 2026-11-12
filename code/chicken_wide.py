# 치킨 표본 정의 넓히기: chicken_check.py(좁은 정의)와 같은 분석을 정의 3단계로 반복해 결과가 정의에 얼마나 흔들리는지 본다.
#  A 좁음  = chicken_check.py 그대로: 업태 '통닭(치킨)' | ('호프/통닭' & 상호에 치킨·통닭·닭강정) | 브랜드 매칭
#  B 넓음  = 업태 무관(일반+휴게): 상호 키워드·브랜드 매칭 | 업태 '통닭(치킨)'. 닭갈비·삼계탕·찜닭 등 비치킨 닭요리, 코리안BBQ·맘스터치 제외
#  C 최대  = B + 상호와 무관하게 업태 '호프/통닭' 전부(치킨 파는 호프 포함, 순수 호프도 섞임 → 민감도용)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl').copy()
a['open']=pd.to_datetime(a['인허가일자'].astype(str).str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].astype(str).str.strip(),errors='coerce')
a['nm']=a['사업장명'].fillna('').str.replace(r'\s+','',regex=True)
src=open(f'{O}/code/wide.py',encoding='utf-8').read(); B=eval(src[src.index('B={')+2:src.index('\nR=[]')])['치킨']
B.update({'깐부':'깐부치킨','바른치킨':'바른치킨','땅땅':'땅땅치킨','둘둘':'둘둘치킨','가마로강정':'가마로','훌랄라':'훌랄라','부어치킨':'부어치킨',
          '오븐에빠진닭':'오븐에빠진닭','KFC':'(?i)KFC|케이에프씨','치킨플러스':'치킨플러스','꾸브라꼬':'꾸브라꼬','맥시칸':'맥시칸','썬더치킨':'썬더치킨'})
a['브랜드']=None
for b,p in B.items(): a.loc[a.nm.str.contains(p,regex=True)&a.브랜드.isna(),'브랜드']=b
KW='치킨|통닭|닭강정|후라이드|프라이드|양념반|순살'
NOT='코리안BBQ|바베큐|비비큐보쌈|맘스터치|닭갈비|삼계|닭한마리|찜닭|닭발|닭곰탕|닭칼국수|닭꼬치|백숙|닭도리'
kw=a.nm.str.contains(KW,regex=True); bad=a.nm.str.contains(NOT,regex=True)&a.브랜드.isna(); br=a.브랜드.notna()
defs={'A_좁음':(a.src=='rest')&((a.업태구분명=='통닭(치킨)')|((a.업태구분명=='호프/통닭')&a.nm.str.contains('치킨|통닭|닭강정'))|br)&~(a.nm.str.contains('코리안BBQ|바베큐|비비큐보쌈')&~br),
      'B_넓음':(kw|br|(a.업태구분명=='통닭(치킨)'))&~bad,
      'C_최대':((kw|br|(a.업태구분명=='통닭(치킨)'))&~bad)|(a.업태구분명=='호프/통닭')}

def run(name,mask):
    c=a[mask].copy(); c['X']=pd.to_numeric(c['좌표정보(X)'],errors='coerce'); c['Y']=pd.to_numeric(c['좌표정보(Y)'],errors='coerce')
    c=c[c.X.notna()].reset_index(drop=True); c['프랜차이즈']=c.브랜드.notna()
    T=cKDTree(c[['X','Y']].values); same=np.zeros(len(c),bool)   # 같은 자리(30m) 폐업 -60일~+90일 안 같은 정의 점포 재등록 = 점주 교체 (11/12 1년→90일: 3개월 넘게 비었다 들어온 건 양도가 아니라 폐업 후 새 세입자)
    for i,r in c[c.close.notna()].iterrows():
        q=[j for j in T.query_ball_point([r.X,r.Y],30) if j!=i]
        if q: o=c.open.values[q]; same[i]=((o>=np.datetime64(r.close-pd.Timedelta(days=60)))&(o<=np.datetime64(r.close+pd.Timedelta(days=90)))).any()
    c['재등록']=same; STORES[name]=c   # 전 기간 점포(재등록 표시 포함)를 다른 스크립트가 쓰도록 보관
    d=c[c.open.dt.year.between(2015,2022)].copy()
    d['y_신고']=(d.close<=d.open+pd.DateOffset(years=3)).astype(int); d['y_실제']=(d.y_신고.astype(bool)&~d.재등록).astype(int)
    ao=c.open.values; ac=c.close.fillna(pd.Timestamp('2100-01-01')).values; dens=[]
    for r in d.itertuples():
        q=np.array(T.query_ball_point([r.X,r.Y],500),dtype=int); t=np.datetime64(r.open); dens.append(((ao[q]<t)&(ac[q]>t)).sum())
    d['밀도']=dens; d['연']=d.open.dt.year; d['면적']=np.log1p(pd.to_numeric(d['소재지면적'],errors='coerce').fillna(40))
    out=dict(정의=name,개업15_22=len(d),프랜차이즈=int(d.프랜차이즈.sum()),휴게=int((d.src=='cafe').sum()),
             폐업신고3y=d.y_신고.mean(),실제폐업3y=d.y_실제.mean(),
             실제_프랜=d[d.프랜차이즈].y_실제.mean(),실제_개인=d[~d.프랜차이즈].y_실제.mean(),밀도중앙=d.밀도.median())
    m=smf.logit('y_실제~np.log1p(밀도)+프랜차이즈+면적+C(연)',d).fit(disp=0)
    out.update(밀도OR=np.exp(m.params['np.log1p(밀도)']),밀도p=m.pvalues['np.log1p(밀도)'],프랜OR=np.exp(m.params['프랜차이즈[T.True]']))
    tr=d.연<=2019; te=~tr; m=smf.logit('y_실제~np.log1p(밀도)+프랜차이즈+면적',d[tr]).fit(disp=0)
    out.update(AUC_프랜만=roc_auc_score(d[te].y_실제,d[te].프랜차이즈.map(d[tr].groupby('프랜차이즈').y_실제.mean())),AUC_모델=roc_auc_score(d[te].y_실제,m.predict(d[te])))
    return out,d

R=[];D={};STORES={}
for k,v in defs.items(): o,D[k]=run(k,v); R.append(o)
pd.set_option('display.width',250)
print(pd.DataFrame(R).set_index('정의').T.round(3).to_string())
d=D['B_넓음']; print('\n[B_넓음] 업태 구성 상위'); print(d.groupby(['src','업태구분명']).size().sort_values(ascending=False).head(8).to_string())
g=d[d.프랜차이즈].groupby('브랜드').agg(n=('y_실제','size'),실제폐업3y=('y_실제','mean'),신고3y=('y_신고','mean')).query('n>=30').sort_values('실제폐업3y')
print('\n[B_넓음] 브랜드별 2015–22 개업, 3년 실제 폐업률 (n>=30)'); print(g.round(3).to_string())
