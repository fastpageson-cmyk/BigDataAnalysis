# 상권 추정매출·상권변화지표(OpenAPI, 2021Q1~)를 개업 직전 분기 값으로 점포에 붙여 폐업 보조지표로 검증
#  표본: 치킨 B_넓음 중 공식 상권 안, 2021Q2~2023Q3 개업 (인허가 자료가 2026-11-04까지라 3년 폐업 관측 가능한 끝)
#  y = 3년 안 실제 폐업(같은 자리 재등록 제외). 통제: 브랜드 과거 폐업률(2010–19 개업) + 면적 + 브랜드 업력 + 개업연도
#  지표: 상권 치킨 매출(점포당)·매출 증감·시간대/요일/연령 구성·외식 중 치킨 비중, 상권변화지표 등급·운영/폐업 영업개월
#  모델 비교: 같은 표본 반복 5겹 교차검증으로 기본 3변수 vs 기본+채택 지표 AUC
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/상권_API'
def load(name):
    spec=importlib.util.spec_from_file_location(name,f'{O}/code/{name}.py'); m=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign')
import sys; H=int(sys.argv[1]) if len(sys.argv)>1 else 3     # 폐업 기준 년수. 2를 주면 2년 폐업·2021Q2–2024Q3 개업으로 표본 확대(민감도)
END={3:'2023-09-30',2:'2024-09-30'}[H]
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year; c['브랜드']=c.브랜드.fillna('개인')
c['y']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)   # 브랜드 이력용(2010–19 개업)은 항상 3년
gi,_,_=aa.assign(c.X.values,c.Y.values); c['TRDAR_CD']=np.where(gi>=0,pd.to_numeric(aa.A.상권코드).values[np.clip(gi,0,None)],-1)
c['q']=c.open.dt.year*10+c.open.dt.quarter
pq=lambda q:(q-1) if q%10>1 else (q//10-1)*10+4                       # 직전 분기
c['q_1']=c.q.map(pq); c['q_5']=c.q_1.map(lambda q:q-10)             # 직전 분기, 그 1년 전
co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values

# ---- 상권 값 ----
s=pd.read_csv(f'{R}/추정매출_상권_외식_2021_.csv'); i=pd.read_csv(f'{R}/상권변화지표_상권_2021_.csv')
ck=s[s.SVC_INDUTY_CD=='CS100007'].set_index(['TRDAR_CD','STDR_YYQU_CD'])
food=s.groupby(['TRDAR_CD','STDR_YYQU_CD']).THSMON_SELNG_AMT.sum()
ix=i.set_index(['TRDAR_CD','STDR_YYQU_CD'])
d=c[(c.open>='2021-04-01')&(c.open<=END)&(c.TRDAR_CD>=0)].copy()
d['y']=((d.close<=d.open+pd.DateOffset(years=H))&~d.재등록).astype(int)
def g(tab,col,qc): return [tab[col].get((t,q),np.nan) if (t,q) in tab.index else np.nan for t,q in zip(d.TRDAR_CD,d[qc])]
amt=np.array(g(ck,'THSMON_SELNG_AMT','q_1'),float)
# 같은 상권 안 개업 시점 영업 중 치킨집 수(인허가) → 점포당 매출
d['상권치킨수']=[((c.TRDAR_CD.values==t)&(co<o)&(cc>o)).sum() for t,o in zip(d.TRDAR_CD,d.open.values)]
d['치킨매출있음']=(~np.isnan(amt)).astype(int)
d['l상권치킨매출']=np.log1p(np.nan_to_num(amt))
d['l점포당매출']=np.log1p(np.nan_to_num(amt)/(d.상권치킨수+1))
a5=np.array(g(ck,'THSMON_SELNG_AMT','q_5'),float); d['치킨매출_1년증감']=np.log1p(np.nan_to_num(amt))-np.log1p(np.nan_to_num(a5))
for nm,cols in {'저녁비중':['TMZON_17_21_SELNG_AMT','TMZON_21_24_SELNG_AMT'],'심야비중':['TMZON_21_24_SELNG_AMT'],'주말비중':['WKEND_SELNG_AMT'],
                '20대비중':['AGRDE_20_SELNG_AMT'],'60대이상비중':['AGRDE_60_ABOVE_SELNG_AMT'],'여성비중':['FML_SELNG_AMT']}.items():
    num=sum(np.array(g(ck,cn,'q_1'),float) for cn in cols); d[nm]=num/np.where(amt>0,amt,np.nan)
d['건당매출']=amt/np.array(g(ck,'THSMON_SELNG_CO','q_1'),float)
fq=np.array([food.get((t,q),np.nan) for t,q in zip(d.TRDAR_CD,d.q_1)],float); f5=np.array([food.get((t,q),np.nan) for t,q in zip(d.TRDAR_CD,d.q_5)],float)
d['외식중치킨비중']=np.nan_to_num(amt)/fq; d['l외식매출']=np.log1p(fq); d['외식매출_1년증감']=np.log1p(fq)-np.log1p(f5)
d['변화지표']=g(ix,'TRDAR_CHNGE_IX','q_1')
for v in ['OPR_SALE_MT_AVRG','CLS_SALE_MT_AVRG','SU_OPR_SALE_MT_AVRG','SU_CLS_SALE_MT_AVRG']: d[v]=g(ix,v,'q_1')
d['운영개월_상대']=d.OPR_SALE_MT_AVRG/d.SU_OPR_SALE_MT_AVRG; d['폐업개월_상대']=d.CLS_SALE_MT_AVRG/d.SU_CLS_SALE_MT_AVRG
for k_ in ['LL','LH','HL','HH']: d['지표_'+k_]=(d.변화지표==k_).astype(int)

# ---- 통제 변수 (indicators.py와 같은 정의, 브랜드 이력만 2010–19 개업으로) ----
h=c[c.연.between(2010,2019)]; gb=h.groupby('브랜드').y.agg(['sum','size']); br=(gb['sum']+20*h.y.mean())/(gb['size']+20)
d['브랜드률']=d.브랜드.map(br).fillna(h.y.mean()); d['l브랜드률']=np.log(d.브랜드률/(1-d.브랜드률))
d['면적']=np.log1p(pd.to_numeric(d['소재지면적'],errors='coerce').fillna(40))
first=c[c.브랜드!='개인'].groupby('브랜드').open.min(); d['브랜드업력']=np.where(d.브랜드=='개인',0,(d.open-d.브랜드.map(first)).dt.days/365)

IND={'상권 치킨 매출(log)':'l상권치킨매출','상권 치킨 점포당 매출(log)':'l점포당매출','상권 치킨 매출 1년 증감':'치킨매출_1년증감','치킨 매출 저녁(17–24시) 비중':'저녁비중',
     '치킨 매출 심야(21–24시) 비중':'심야비중','치킨 매출 주말 비중':'주말비중','치킨 매출 20대 비중':'20대비중','치킨 매출 60대+ 비중':'60대이상비중','치킨 매출 여성 비중':'여성비중',
     '치킨 결제 건당 매출':'건당매출','외식 매출 중 치킨 비중':'외식중치킨비중','상권 외식 매출(log)':'l외식매출','상권 외식 매출 1년 증감':'외식매출_1년증감',
     '상권 치킨집 수(인허가)':'상권치킨수','운영 영업개월(상권/서울)':'운영개월_상대','폐업 영업개월(상권/서울)':'폐업개월_상대',
     '변화지표 LL(다이나믹)':'지표_LL','변화지표 LH(상권확장)':'지표_LH','변화지표 HL(상권축소)':'지표_HL','변화지표 HH(정체)':'지표_HH'}
BASE='l브랜드률+면적+브랜드업력+C(연)'
if __name__=='__main__':
    print(f'[{H}년 폐업] 표본: 공식 상권 안 2021Q2–{END[:7]} 개업 {len(d)}곳 (프랜차이즈 {(d.브랜드!="개인").sum()}), {H}년 실제 폐업률 {d.y.mean():.3f}, 직전 분기 치킨 매출 있음 {d.치킨매출있음.mean():.2f}')
    print('변화지표 분포·폐업률:',d.groupby('변화지표').y.agg(['size','mean']).round(3).to_dict('index'))
    rows=[]
    for nm,col in IND.items():
        x=d.dropna(subset=[col]).copy(); x=x[np.isfinite(x[col])]
        if col.startswith(('l점포당','치킨매출_1년','저녁','심야','주말','20대','60대','여성','건당','외식중')): x=x[x.치킨매출있음==1]
        x['z']=(x[col]-x[col].mean())/x[col].std(); m=smf.logit(f'y~{BASE}+z',x).fit(disp=0)
        bin_=x[col].nunique()<=2; hi=x[col]==1 if bin_ else x[col]>=x[col].quantile(.8); lo=x[col]==0 if bin_ else x[col]<=x[col].quantile(.2)
        rows.append(dict(지표=nm,n=len(x),하위_폐업률=x[lo].y.mean(),상위_폐업률=x[hi].y.mean(),OR_1SD=np.exp(m.params['z']),p=m.pvalues['z']))
    r=pd.DataFrame(rows); r['q_BH']=multipletests(r.p,method='fdr_bh')[1]
    pd.set_option('display.width',250); print('\n[지표별] 통제: 브랜드 과거 폐업률+면적+브랜드 업력+개업연도. 하위/상위 = 하위·상위 20%(이진은 0 vs 1)')
    print(r.sort_values('p').round(3).to_string(index=False)); r.to_csv(f'{O}/data/processed/trdar_tests_{H}y.csv',index=False,encoding='utf-8-sig')

    # 모델 비교: 반복 5겹 CV (같은 접힘으로 짝 비교)
    sel=r[r.p<.1].지표.map(IND).tolist(); print('\n후보(p<0.1, 사후 선택이라 낙관적):',sel)
    X=d[['l브랜드률','면적','브랜드업력']+[v for v in IND.values()]].copy(); X=X.replace([np.inf,-np.inf],np.nan); X=X.fillna(X.median())
    yy=d.y.values; cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=20,random_state=0); A={'기본 3변수':[], '기본+p<0.1 지표':[], '기본+상권 지표 전부(L2)':[]}
    specs={'기본 3변수':['l브랜드률','면적','브랜드업력'],'기본+p<0.1 지표':['l브랜드률','면적','브랜드업력']+sel,'기본+상권 지표 전부(L2)':list(X.columns)}
    for tr,te in cv.split(X,yy):
        for k,cols in specs.items():
            Z=X[cols]; mu,sd=Z.iloc[tr].mean(),Z.iloc[tr].std().replace(0,1)
            m=LogisticRegression(C=1.0 if k!='기본+상권 지표 전부(L2)' else 0.1,max_iter=3000).fit((Z.iloc[tr]-mu)/sd,yy[tr]); A[k].append(roc_auc_score(yy[te],m.predict_proba((Z.iloc[te]-mu)/sd)[:,1]))
    A={k:np.array(v) for k,v in A.items()}; b=A['기본 3변수']
    for k,v in A.items(): print(f'  {k}: CV AUC {v.mean():.3f} (차이 vs 기본 {np.mean(v-b):+.3f}, 접힘별 차이>0 비율 {(v-b>0).mean():.2f})')
    d.drop(columns=[x for x in d.columns if x in('지번주소','사업장명','nm','도로명주소')]).to_pickle(f'{O}/data/processed/_trdar_sample.pkl')
