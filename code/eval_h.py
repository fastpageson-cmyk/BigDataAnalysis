# 기간 h(1~7년)별: 기준선② 브랜드평균 vs 모델, 브랜드 내부 쌍 AUC. 학습=앞 코호트, 평가=관측 가능한 최근 2년 코호트
import os,sys,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score,average_precision_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
s=pd.read_pickle(f'{O}/data/processed/store_feats.pkl')
COMP=['밀도300','휴게밀도300','폐업1y500','신규1y500','같은브랜드500','같은브랜드1km','같은업종타브랜드500','같은업종신규1y500','최근접같은브랜드m']
STORE=['면적','휴게','다중','분기']
DL=['동_음식점수','동_3y증가','동_폐업률','동_개업률','동_카페비중']
DM=['동_외식점포','동_점포당매출','동_유동인구','동_직장인구','동_상주인구','동_집객시설']
def lab(d,h,ycol): return ((d.close<=d.open+pd.DateOffset(years=h))&(d[ycol] if ycol else True)).astype(int)
def pairauc(y,p,g):
    # 같은 브랜드 안의 (폐점, 생존) 쌍에서 폐점 점포 점수가 더 높은 비율
    num=den=0
    for b in np.unique(g):
        m=g==b; yy=y[m]; pp=p[m]; e=pp[yy==1]; n=pp[yy==0]
        if len(e) and len(n): num+=((e[:,None]>n[None]).sum()+0.5*(e[:,None]==n[None]).sum()); den+=len(e)*len(n)
    return num/den if den else np.nan
def prep(d,tr,F):
    X=pd.get_dummies(d[F],columns=[c for c in F if not pd.api.types.is_numeric_dtype(d[c])],dtype=float); X=X.fillna(X[tr].median()); return (X-X[tr].mean())/X[tr].std().replace(0,1)
def run(h,ycol,F,test_years=2,train_years=6,label=''):
    last=(END-pd.DateOffset(years=h)); te_lo=last-pd.DateOffset(years=test_years); tr_lo=te_lo-pd.DateOffset(years=train_years)
    d=s[(s.open>tr_lo)&(s.open<=last)].copy(); y=lab(d,h,ycol).values; tr=(d.open<=te_lo).values; te=~tr
    br=pd.Series(y[tr]).groupby(d.브랜드.values[tr]).agg(['sum','count']); pr=y[tr].mean(); enc=(br['sum']+20*pr)/(br['count']+20)
    d['브랜드률']=d.브랜드.map(enc).fillna(pr)
    base=d.브랜드률.values[te]+1e-9*np.random.RandomState(0).rand(te.sum()); g=d.브랜드.values[te]; yt=y[te]
    Z=prep(d,tr,['브랜드률','업종']+F); lr=LogisticRegression(C=0.1,max_iter=3000).fit(Z[tr],y[tr]); p=lr.predict_proba(Z[te])[:,1]
    Zb=prep(d,tr,['브랜드']+F); lb=LogisticRegression(C=0.1,max_iter=3000).fit(Zb[tr],y[tr]); pw=lb.predict_proba(Zb[te])[:,1]
    gb=HGB(max_depth=3,learning_rate=0.05,max_iter=200,random_state=0).fit(Z[tr],y[tr]); pg=gb.predict_proba(Z[te])[:,1]
    k=max(1,int(te.sum()*.1)); top=lambda q: yt[np.argsort(-q)[:k]].mean()
    return dict(설정=label,h=h,학습=f'{tr_lo:%Y-%m}~{te_lo:%Y-%m}',평가=f'~{last:%Y-%m}',n평가=int(te.sum()),사건평가=int(yt.sum()),사건률=round(yt.mean(),3),
        기준선AUC=round(roc_auc_score(yt,base),3),로지AUC=round(roc_auc_score(yt,p),3),GBM_AUC=round(roc_auc_score(yt,pg),3),
        기준선top10=round(top(base),3),로지top10=round(top(p),3),GBMtop10=round(top(pg),3),
        브랜드내부쌍AUC=round(pairauc(yt,pw,g),3))
if __name__=='__main__':
    R=[]
    for h in range(1,8):
        for yc,nm in [('진짜폐점','실제폐점'),(None,'인허가폐업')]:
            R.append(run(h,yc,STORE+COMP+DL,label=nm))
    r=pd.DataFrame(R); pd.set_option('display.width',260); print(r.to_string(index=False))
    r.to_csv(f'{O}/data/processed/eval_horizon.csv',index=False,encoding='utf-8-sig')
