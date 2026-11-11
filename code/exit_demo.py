# 출구별 예측 데모: rec_demo의 5년 궁합 모델을 출구 좌표에 적용(상권 성격은 출구가 속한 상권, 경쟁 변수는 출구 지점 기준)
import io,contextlib,os,numpy as np,pandas as pd
with contextlib.redirect_stdout(io.StringIO()): import rec_demo as R
from area_assign import assign
O=R.O; ex=pd.read_csv(f'{O}/data/processed/exits.csv')
out=[]
for st in ['강남','홍대입구','여의도']:
    E=ex[ex.역명==st].copy(); gi,_,_=assign(E.X.values,E.Y.values); E['gi']=gi
    for r in E.itertuples():
        if r.gi<0: continue
        a=R.A.iloc[r.gi]; 
        if a.상권코드 not in R.P.index: continue
        pf=R.P.loc[a.상권코드]; sub=R.s[R.s.상권코드==a.상권코드]; adm=sub.adm.mode().iloc[0]; dv=R.dl[(R.dl.adm==adm)&(R.dl.Y==2025)].iloc[0]
        for c,b in R.brands.itertuples(index=False):
            if c not in('커피','치킨'): continue
            f=R.comp(r.X,r.Y,b,c); f.update({k:pf[k] for k in R.PROF}); f.update({k:dv[k] for k in R.DL}); f['동_음식점수']=np.log1p(f['동_음식점수'])
            f.update(dict(면적=np.log1p(55),휴게=int(c=='커피'),다중=0,브랜드=b,업종=c,브랜드률=R.enc[b],역=st,출구=int(r.출구),상권=a.상권명)); out.append(f)
D=pd.DataFrame(out); D['위험5y']=R.m.predict_proba(R.zi(R.design(D),D))[:,1]
D.to_csv(f'{O}/data/processed/exit_demo.csv',index=False,encoding='utf-8-sig')
for st in ['강남','홍대입구','여의도']:
    for c in ['커피','치킨']:
        t=D[(D.역==st)&(D.업종==c)]; p=t.pivot_table(index='브랜드',columns='출구',values='위험5y')
        rk=t.assign(r=t.groupby('출구').위험5y.rank()).pivot_table(index='브랜드',columns='출구',values='r')
        top=t.loc[t.groupby('출구').위험5y.idxmin()].set_index('출구').브랜드.to_dict()
        print(f'\n## {st}역 {c}: 출구별 1위 {top}'); print('  브랜드별 출구간 위험 범위(최소~최대):',{b:f'{v.min():.3f}~{v.max():.3f}' for b,v in p.iterrows()})
        print('  같은브랜드500 출구별(메가/교촌):',t[t.브랜드.isin(['메가커피','교촌'])].pivot_table(index='브랜드',columns='출구',values='같은브랜드500').astype(int).to_dict('index'))
