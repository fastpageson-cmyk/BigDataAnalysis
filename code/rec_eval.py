# 추천 검증: 같은 상권·같은 업종에 개업한 서로 다른 브랜드 쌍에서, 먼저 망한 쪽을 더 위험하다고 예측했나(쌍 정확도)
#  비교: 기준선(브랜드 평균) / 가산 모델(브랜드+입지) / 궁합 모델(브랜드×상권 성격 상호작용 GBM)
import os,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl').join(pd.read_pickle(f'{O}/data/processed/area_profile.pkl'),on='상권코드')
PROF=['외식매출(log)','주말비중','점심비중','야간비중','20대비중','30·40대비중','50대+비중','여성비중']
COMP=['밀도300','휴게밀도300','폐업1y500','신규1y500','같은브랜드500','같은브랜드1km','같은업종타브랜드500','같은업종신규1y500','최근접같은브랜드m']
DL=['동_음식점수','동_3y증가','동_폐업률','동_개업률','동_카페비중']; ST=['면적','휴게','다중']
def pairs(d,score,y):
    num=den=0
    for (a,g),x in d.groupby(['상권코드','업종']):
        if x.브랜드.nunique()<2: continue
        i=x.index.values; yy=y.loc[i].values; sc=score.loc[i].values; b=x.브랜드.values
        for p in range(len(i)):
            for q in range(len(i)):
                if yy[p]==1 and yy[q]==0 and b[p]!=b[q]: den+=1; num+=(sc[p]>sc[q])+0.5*(sc[p]==sc[q])
    return num/den if den else np.nan,den
for H,yc in [(3,'진짜폐점'),(5,'진짜폐점'),(3,None)]:
    last=END-pd.DateOffset(years=H); te_lo=last-pd.DateOffset(years=3); d=s[(s.open.dt.year>=2008)&(s.open<=last)&s.상권코드.notna()&s['외식매출(log)'].notna()].copy()
    y=((d.close<=d.open+pd.DateOffset(years=H))&(d[yc] if yc else True)).astype(int); tr=d.open<=te_lo; te=~tr
    br=y[tr].groupby(d.브랜드[tr]).agg(['sum','count']); pr=y[tr].mean(); d['브랜드률']=d.브랜드.map((br['sum']+20*pr)/(br['count']+20)).fillna(pr)
    X=pd.concat([d[['브랜드률']+ST+COMP+DL+PROF],pd.get_dummies(d.업종,dtype=float)],axis=1); X=X.fillna(X[tr].median()); Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    add=pd.Series(LogisticRegression(C=0.1,max_iter=3000).fit(Z[tr],y[tr]).predict_proba(Z)[:,1],index=d.index)
    # 궁합: 브랜드 원-핫과 상권 프로필의 곱 항을 GBM이 학습(깊이 4)
    Xb=pd.concat([X,pd.get_dummies(d.브랜드,dtype=float)],axis=1)
    gb=pd.Series(HGB(max_depth=4,learning_rate=0.04,max_iter=300,l2_regularization=1.0,random_state=0).fit(Xb[tr],y[tr]).predict_proba(Xb)[:,1],index=d.index)
    Xi=Z.copy()
    for b in d.브랜드[tr].value_counts().index[:15]:
        bb=(d.브랜드==b).astype(float)
        for c in PROF: Xi[f'{b}×{c}']=bb*Z[c]
    inter=pd.Series(LogisticRegression(C=0.05,max_iter=5000).fit(Xi[tr],y[tr]).predict_proba(Xi)[:,1],index=d.index)
    t=d[te]; base=d.브랜드률+1e-9*np.random.RandomState(0).rand(len(d))
    r={k:pairs(t,v,y) for k,v in [('기준선(브랜드평균)',base),('가산(브랜드+입지)',add),('궁합 GBM',gb),('궁합 로지스틱',inter)]}
    print(f'H={H} {yc or "인허가폐업"} 평가 개업 {te_lo:%Y-%m}~{last:%Y-%m} n={te.sum()} 사건={y[te].sum()} 쌍={r["기준선(브랜드평균)"][1]} |',' | '.join(f'{k} {v[0]:.3f}' for k,v in r.items()))

# H=5 실제폐점: 상권 단위 부트스트랩으로 (궁합 로지스틱 − 기준선) 쌍 정확도 차이 CI
def pair_list(d,score,y):
    R=[]
    for (a,g),x in d.groupby(['상권코드','업종']):
        i=x.index.values; yy=y.loc[i].values; sc=score.loc[i].values; b=x.브랜드.values
        for p in range(len(i)):
            for q in range(len(i)):
                if yy[p]==1 and yy[q]==0 and b[p]!=b[q]: R.append((a,(sc[p]>sc[q])+0.5*(sc[p]==sc[q])))
    return pd.DataFrame(R,columns=['a','hit'])
if True:
    H=5; last=END-pd.DateOffset(years=H); te_lo=last-pd.DateOffset(years=3); d=s[(s.open.dt.year>=2008)&(s.open<=last)&s.상권코드.notna()&s['외식매출(log)'].notna()].copy()
    y=((d.close<=d.open+pd.DateOffset(years=H))&d.진짜폐점).astype(int); tr=d.open<=te_lo; te=~tr
    br=y[tr].groupby(d.브랜드[tr]).agg(['sum','count']); pr=y[tr].mean(); d['브랜드률']=d.브랜드.map((br['sum']+20*pr)/(br['count']+20)).fillna(pr)
    X=pd.concat([d[['브랜드률']+ST+COMP+DL+PROF],pd.get_dummies(d.업종,dtype=float)],axis=1); X=X.fillna(X[tr].median()); Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    for b in d.브랜드[tr].value_counts().index[:15]:
        bb=(d.브랜드==b).astype(float)
        for c in PROF: Z[f'{b}×{c}']=bb*Z[c]
    m=LogisticRegression(C=0.05,max_iter=5000).fit(Z[tr],y[tr]); inter=pd.Series(m.predict_proba(Z)[:,1],index=d.index)
    base=d.브랜드률+1e-9*np.random.RandomState(0).rand(len(d))
    A=pair_list(d[te],base,y).rename(columns={'hit':'b'}); B=pair_list(d[te],inter,y); A['i']=B.hit.values
    g=A.groupby('a')[['b','i']].agg(['sum','count']); rs=np.random.RandomState(0); keys=g.index.values; D=[]
    for _ in range(2000):
        k=rs.choice(keys,len(keys)); x=g.loc[k]; D.append(x[('i','sum')].sum()/x[('i','count')].sum()-x[('b','sum')].sum()/x[('b','count')].sum())
    print('H=5 궁합−기준선 쌍정확도 차',round(A.i.mean()-A.b.mean(),3),'95%CI',np.percentile(D,[2.5,97.5]).round(3),'상권 수',len(keys))
    co=pd.Series(m.coef_[0],index=Z.columns); ix=co[co.index.str.contains('×')].sort_values(); print('궁합 계수 하위(덜 망함)',ix.head(6).round(2).to_dict()); print('상위(더 망함)',ix.tail(6).round(2).to_dict())
