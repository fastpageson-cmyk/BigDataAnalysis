# brand_fit.py에서 사후에 찾은 브랜드×입지 패턴 4개 검증
#  주장1 bhc·BBQ 발달상권 폐업↑  주장2 BBQ 주거지 < 먹자골목  주장3 푸라닭 주거지 < 먹자골목  주장4 교촌·지코바 어디서나 낮음
#  A. 상호 확인: 해당 칸 점포가 같은 매장 형태인지(카페형·주점형 등 다른 포맷 섞임)
#  B. 반반 재현: 2010–15 개업 / 2016–21 개업 따로
#  C. 새 코호트: 2022–23 개업, 3년 실제 폐업(앞 분석에 안 쓴 점포)
#  D. 통제 추가: 면적·연도·500m 경쟁, 2022Q3 일괄 말소 의심 폐업 제외
#  E. 다중비교: 브랜드 안에서 입지 라벨을 섞는 순열 1,000회, '모든 브랜드×유형 칸 중 최대 차이'와 비교(가족 단위 p)
import os,io,contextlib,importlib.util,re,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from scipy.spatial import cKDTree
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec=importlib.util.spec_from_file_location('bf',f'{O}/code/brand_fit.py'); bf=importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(bf)
c,f=bf.c,bf.f.copy()
CT=cKDTree(c[['X','Y']].values); co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values

def claims(g,y='y'):
    """주장별 (비교 A 폐업률, 비교 B 폐업률, nA, nB)"""
    hb=g[g.브랜드.isin(['bhc','BBQ'])]; out={}
    out['1 bhc·BBQ 발달 vs 그외']=(hb[hb.공식유형=='발달상권'][y],hb[hb.공식유형!='발달상권'][y])
    for b,k in [('BBQ','2 BBQ'),('푸라닭','3 푸라닭')]:
        s=g[g.브랜드==b]; out[f'{k} 먹자골목 vs 주거지']=(s[s.사람유형=='먹자골목'][y],s[s.사람유형=='주거지'][y])
    return {k:(a.mean(),b.mean(),len(a),len(b)) for k,(a,b) in out.items()}
show=lambda r:pd.DataFrame(r,index=['A 폐업률','B 폐업률','nA','nB']).T.round(3)

print('[원래 결과] 2010–21 개업, 5년'); print(show(claims(f)).to_string())

# A. 상호 형태
print('\n[A] 상호에 들어간 매장 형태 키워드 (해당 칸 점포 수)')
KW={'카페':'카페|cafe|CAFE','비어/호프':'비어|호프|BEER|펍|PUB','프리미엄':'프리미엄|premium','치킨':'치킨'}
for b in ['bhc','BBQ','푸라닭']:
    for v,lv in [('공식유형','발달상권'),('사람유형','먹자골목'),('사람유형','주거지')]:
        s=f[(f.브랜드==b)&(f[v]==lv)]
        if len(s)<15: continue
        cnt={k:int(s.nm.str.contains(p,case=False,regex=True).sum()) for k,p in KW.items()}
        cl={k:round(s[s.nm.str.contains(p,case=False,regex=True)].y.mean(),2) for k,p in KW.items() if s.nm.str.contains(p,case=False,regex=True).sum()>=5}
        print(f'  {b} {lv} n={len(s)}: {cnt}  키워드별 폐업률(5곳 이상) {cl}')

# B. 반반 재현
print('\n[B] 반반 재현');
for nm,g in [('2010–15',f[f.연<=2015]),('2016–21',f[f.연>=2016])]: print(f'-- {nm}'); print(show(claims(g)).to_string())

# C. 새 코호트 2022–23, 3년
n=c[c.프랜차이즈&(c.open>=pd.Timestamp('2022-01-01'))&(c.open<=pd.Timestamp('2023-11-04'))].copy()
n['y3']=((n.close<=n.open+pd.DateOffset(years=3))&~n.재등록).astype(int)
print(f'\n[C] 새 코호트 2022-01~2023-11 개업 프랜차이즈 {len(n)}곳, 3년 실제 폐업 (앞 분석에 안 쓴 점포)'); print(show(claims(n,'y3')).to_string())

# D. 통제 추가 + 2022Q3 제외
f['경쟁']=[((co[q]<o)&(cc[q]>o)).sum() for q,o in zip(CT.query_ball_point(f[['X','Y']].values,500),f.open.values)]
q3=f.close.between(pd.Timestamp('2022-07-01'),pd.Timestamp('2022-09-30'))
print('\n[D] 로지스틱(면적·연도·log 경쟁 통제), 2022Q3 폐업 제외 버전 함께')
for lab,g in [('전체',f),('2022Q3 제외',f[~q3])]:
    hb=g[g.브랜드.isin(['bhc','BBQ'])].assign(t=lambda x:x.공식유형=='발달상권')
    m=smf.logit('y~C(브랜드)+면적+C(연)+np.log1p(경쟁)+t',hb).fit(disp=0); r=[f'주장1 OR {np.exp(m.params["t[T.True]"]):.2f} p={m.pvalues["t[T.True]"]:.4f}']
    for b,k in [('BBQ','주장2'),('푸라닭','주장3')]:
        s=g[(g.브랜드==b)&g.사람유형.isin(['먹자골목','주거지'])].assign(t=lambda x:x.사람유형=='먹자골목')
        m=smf.logit('y~면적+연+np.log1p(경쟁)+t',s).fit(disp=0); r.append(f'{k} OR {np.exp(m.params["t[T.True]"]):.2f} p={m.pvalues["t[T.True]"]:.4f}')
    print(f'  {lab}: '+' | '.join(r))

# E. 다중비교: 모든 브랜드 × (유형 vs 나머지) 칸의 차이 z 중 최댓값 분포
def maxz(g):
    zs={}
    for v in ['공식유형','사람유형']:
        for b,s in g.groupby('브랜드'):
            p0=s.y.mean()
            for lv,x in s.groupby(v):
                r=s[s[v]!=lv]
                if len(x)<15 or len(r)<15 or p0 in(0,1): continue
                se=np.sqrt(p0*(1-p0)*(1/len(x)+1/len(r))); zs[(b,v,lv)]=(x.y.mean()-r.y.mean())/se
    return zs
obs=maxz(f); rng=np.random.default_rng(1); mx=[]
for _ in range(1000):
    g=f.copy()
    for v in ['공식유형','사람유형']: g[v]=g.groupby('브랜드')[v].transform(lambda x:rng.permutation(x.values))
    mx.append(max(abs(z) for z in maxz(g).values()))
mx=np.array(mx)
print(f'\n[E] 다중비교: 비교한 브랜드×유형 칸 {len(obs)}개, 순열 1,000회의 최대 |z| 분포와 비교한 가족 단위 p')
for k in [('BBQ','공식유형','발달상권'),('bhc','공식유형','발달상권'),('BBQ','사람유형','주거지'),('BBQ','사람유형','먹자골목'),('푸라닭','사람유형','먹자골목'),('푸라닭','사람유형','주거지')]:
    if k in obs: print(f'  {k}: z={obs[k]:.2f}, 가족 단위 p={(np.sum(mx>=abs(obs[k]))+1)/1001:.3f}')
top=sorted(obs.items(),key=lambda x:-abs(x[1]))[:5]; print('  관측 |z| 상위 5:',[(k,round(z,2)) for k,z in top])

# 주장4: 교촌·지코바 유형별 95% 신뢰구간
print('\n[주장4] 교촌·지코바 유형별 5년 실제 폐업률 [95% 신뢰구간]')
for b in ['교촌','지코바']:
    for v in ['공식유형','사람유형']:
        s=f[f.브랜드==b].groupby(v).y.agg(['sum','size'])
        print(f'  {b} {v}: '+', '.join(f'{lv} {r["sum"]}/{r["size"]} [{proportion_confint(r["sum"],r["size"],method="wilson")[0]:.2f}–{proportion_confint(r["sum"],r["size"],method="wilson")[1]:.2f}]' for lv,r in s.iterrows()))
n4=n[n.브랜드.isin(['교촌','지코바'])]; print(f'  새 코호트 2022–23 교촌·지코바 3년 실제 폐업 {n4.y3.sum()}/{len(n4)}')

# 참고: 발표 메시지 후보 '프랜차이즈는 주거지에서 덜 망함'(brand_fit [3], p=0.002)도 같은 방식으로 재현 확인
print('\n[참고] 프랜차이즈 사람유형별 폐업률 재현')
for nm,g,y in [('2010–15, 5년',f[f.연<=2015],'y'),('2016–21, 5년',f[f.연>=2016],'y'),('새 코호트 2022–23, 3년',n,'y3')]:
    print(f'  {nm}: '+', '.join(f'{k} {v:.3f}({m})' for (k,v),m in zip(g.groupby("사람유형")[y].mean().items(),g.groupby("사람유형").size())))
