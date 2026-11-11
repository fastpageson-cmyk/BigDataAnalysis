# 브랜드×입지 궁합: "입지로 폐업을 예측하나"가 아니라 "브랜드마다 잘 맞는 입지가 다른가"를 본다 (치킨, chicken_wide.py B_넓음)
#  표본: 2010–21 개업 프랜차이즈(서울 점포 40곳 이상 브랜드), y = 5년 안 실제 폐업
#  입지 변수(개업 시점): 공식 상권 유형(골목·발달·전통시장·상권 밖), 사람·이동 유형(location_clusters.py KMeans), 500m 경쟁, 점포 면적
#  1) 변수마다 브랜드×입지 상호작용 우도비검정 + 순열검정(입지 라벨을 브랜드 안에서 섞어 우연 분포)
#  2) 브랜드별 입지 칸 폐업률표(점포 15곳 이상 칸)
#  3) 입지 변수 자체의 의미: 개인 vs 프랜차이즈에서 각 변수 효과
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from scipy.stats import chi2
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
def load(name):
    spec=importlib.util.spec_from_file_location(name,f'{O}/code/{name}.py'); m=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign')
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year; c['면적']=np.log1p(pd.to_numeric(c['소재지면적'],errors='coerce').fillna(40))
c['y']=((c.close<=c.open+pd.DateOffset(years=5))&~c.재등록).astype(int)
gi,_,_=aa.assign(c.X.values,c.Y.values)
c['공식유형']=np.where(gi>=0,aa.A.상권유형.values[np.clip(gi,0,None)],'상권밖')
c.loc[c.공식유형=='관광특구','공식유형']='발달상권'   # 관광특구 6곳은 발달상권에 합침
T=pd.read_csv(f'{P}/area_types_people.csv').set_index('TRDAR_CD').유형
code=pd.to_numeric(aa.A.상권코드).values
c['사람유형']=[{0:'직장가',1:'주거지',2:'먹자골목'}.get(T.get(code[i]),'상권밖') if i>=0 else '상권밖' for i in gi]
CT=cKDTree(c[['X','Y']].values); co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values
d=c[c.연.between(2010,2021)].copy()
d['경쟁']=[((co[q]<o)&(cc[q]>o)).sum() for q,o in zip(CT.query_ball_point(d[['X','Y']].values,500),d.open.values)]
d['경쟁구간']=pd.qcut(d.경쟁,3,labels=['경쟁낮음','경쟁중간','경쟁높음']).astype(str)
d['면적구간']=pd.qcut(d.면적,3,labels=['소형','중형','대형']).astype(str)
f=d[d.프랜차이즈]; f=f[f.브랜드.isin(f.브랜드.value_counts()[lambda s:s>=40].index)].copy()
print(f'표본: 2010–21 개업 프랜차이즈 {len(f)}곳, 브랜드 {f.브랜드.nunique()}개, 5년 실제 폐업률 {f.y.mean():.3f}')

def lr(data,v):
    b0=smf.logit(f'y~C(브랜드)+C({v})+C(연)',data).fit(disp=0,method='bfgs',maxiter=3000)
    b1=smf.logit(f'y~C(브랜드)*C({v})+C(연)',data).fit(disp=0,method='bfgs',maxiter=3000)
    return 2*(b1.llf-b0.llf),len(b1.params)-len(b0.params)
rng=np.random.default_rng(0); rows=[]
for v in ['공식유형','사람유형','경쟁구간','면적구간']:
    s,df_=lr(f,v); null=[]
    for _ in range(200):
        g=f.copy(); g[v]=g.groupby('브랜드')[v].transform(lambda x:rng.permutation(x.values)); null.append(lr(g,v)[0])
    rows.append(dict(입지변수=v,χ2=s,df=df_,p_근사=chi2.sf(s,df_),p_순열=(np.sum(np.array(null)>=s)+1)/201))
print('\n[1] 브랜드×입지 궁합 검정 (같은 입지 효과가 브랜드마다 다른가)'); print(pd.DataFrame(rows).round(3).to_string(index=False))

print('\n[2] 브랜드별 입지 칸 5년 실제 폐업률 (점포 15곳 이상 칸, 괄호 n)')
for v in ['공식유형','경쟁구간']:
    t=f.groupby(['브랜드',v]).y.agg(['mean','size']); t=t[t['size']>=15]
    show=(t['mean'].round(2).astype(str)+'('+t['size'].astype(str)+')').unstack().fillna('')
    print(f'-- {v}'); print(show.to_string())

print('\n[3] 입지 변수 자체 효과 (브랜드·연도·면적 통제, 5년 실제 폐업)')
for nm,g,fx in [('개인',d[~d.프랜차이즈],'면적+C(연)'),('프랜차이즈',f,'C(브랜드)+면적+C(연)')]:
    for v in ['공식유형','사람유형','경쟁구간']:
        m0=smf.logit(f'y~{fx}',g).fit(disp=0,method='bfgs',maxiter=2000); m1=smf.logit(f'y~{fx}+C({v})',g).fit(disp=0,method='bfgs',maxiter=2000)
        s=2*(m1.llf-m0.llf); k=len(m1.params)-len(m0.params); rt=g.groupby(v).y.mean().round(3).to_dict()
        print(f'  {nm} {v}: χ²={s:.1f}, p={chi2.sf(s,k):.3f}, 폐업률 {rt}')

print('\n[4] 눈에 띈 패턴 확인')
t=f.groupby(['브랜드','면적구간']).y.agg(['mean','size']); t=t[t['size']>=15]
print('-- 면적구간 (브랜드×면적 궁합이 순열검정에서도 남은 변수)'); print((t['mean'].round(2).astype(str)+'('+t['size'].astype(str)+')').unstack().fillna('')[['소형','중형','대형']].to_string())
t=f.groupby(['브랜드','사람유형']).y.agg(['mean','size']); t=t[t['size']>=15]
print('-- 사람유형'); print((t['mean'].round(2).astype(str)+'('+t['size'].astype(str)+')').unstack().fillna('').to_string())
for bs in (['bhc','BBQ'],):
    g=f[f.브랜드.isin(bs)].assign(발달=lambda x:x.공식유형=='발달상권'); m=smf.logit('y~C(브랜드)+면적+C(연)+발달',g).fit(disp=0)
    print(f'  {"+".join(bs)} 발달상권 vs 그 외: {g[g.발달].y.mean():.3f} vs {g[~g.발달].y.mean():.3f}, OR {np.exp(m.params["발달[T.True]"]):.2f}, p={m.pvalues["발달[T.True]"]:.4f} (사후 발견이라 가설로만)')
