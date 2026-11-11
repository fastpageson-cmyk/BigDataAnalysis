# 파일럿: 개업 시점 정보 / 개업 1년 시점 정보로 폐점 예측, 브랜드 평균 기준선과 비교 (제안서 재검토용, 본분석 아님)
import os,sys,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score,average_precision_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
END=pd.Timestamp('2026-11-04')
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl')
a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a['X']=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); a['Y']=pd.to_numeric(a['좌표정보(Y)'],errors='coerce')
a=a[a.X.notna()&a.open.notna()].reset_index(drop=True)
T=cKDTree(a[['X','Y']].values); ao=a.open.values; ac=a.close.fillna(pd.Timestamp('2100-01-01')).values; asrc=a.src.values
s=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); s=s[s.X.notna()&s.open.dt.year.between(2018,2023)].copy().reset_index(drop=True)
b_open=s.open.values; 
allb=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); allb=allb[allb.X.notna()].reset_index(drop=True)
TB=cKDTree(allb[['X','Y']].values); bo=allb.open.values; bc=allb.close.fillna(pd.Timestamp('2100-01-01')).values; bb=allb.브랜드.values; bg=allb.업종.values
def feats(t_off):
    R=[]
    for i,r in s.iterrows():
        t=np.datetime64(r.open+pd.DateOffset(years=t_off)); t1=t-np.timedelta64(365,'D')
        p=np.array(T.query_ball_point([r.X,r.Y],300)); live=(ao[p]<t)&(ac[p]>t)
        p5=np.array(T.query_ball_point([r.X,r.Y],500)); cl1=((ac[p5]<=t)&(ac[p5]>t1)).sum(); new1=((ao[p5]<=t)&(ao[p5]>t1)).sum()
        q=np.array(TB.query_ball_point([r.X,r.Y],500)); q=q[allb.관리번호.values[q]!=r.관리번호]; lb=(bo[q]<t)&(bc[q]>t)
        R.append(dict(밀도300=live.sum(),휴게밀도300=(live&(asrc[p]=='cafe')).sum(),폐업1y500=cl1,신규1y500=new1,
                      같은브랜드500=(lb&(bb[q]==r.브랜드)).sum(),같은업종타브랜드500=(lb&(bg[q]==r.업종)&(bb[q]!=r.브랜드)).sum(),
                      같은업종신규1y500=((bo[q]<=t)&(bo[q]>t1)&(bg[q]==r.업종)&(bb[q]!=r.브랜드)).sum()))
    return pd.DataFrame(R)
s['면적']=np.log1p(pd.to_numeric(s['소재지면적'],errors='coerce').fillna(55))
s['휴게']=(s.src=='cafe').astype(int); s['다중']=(s['다중이용업소여부']=='Y').astype(int)
s['구']=s['지번주소'].str.extract(r'서울특별시\s+(\S+구)')[0].fillna('NA'); s['연']=s.open.dt.year; s['분기']=s.open.dt.quarter
def run(name,t0,h,ycol,cond):
    d=s[cond].copy(); f=feats(t0).loc[d.index]; d=pd.concat([d,f],axis=1)
    tstart=d.open+pd.DateOffset(years=t0); tend=d.open+pd.DateOffset(years=t0+h)
    d=d[(tend<=END)]
    y=((d.close>tstart.loc[d.index])&(d.close<=tend.loc[d.index])&d[ycol]).astype(int) if ycol else ((d.close>tstart.loc[d.index])&(d.close<=tend.loc[d.index])).astype(int)
    tr=d.연<=2021; te=~tr
    br=y[tr].groupby(d.브랜드[tr]).agg(['sum','count']); prior=y[tr].mean(); k=20
    enc=((br['sum']+k*prior)/(br['count']+k)); d['브랜드률']=d.브랜드.map(enc).fillna(prior)
    num=['브랜드률','면적','휴게','다중','분기']+list(f.columns)
    X=pd.get_dummies(d[num+['구','업종']],columns=['구','업종'],dtype=float)
    out={'n학습':int(tr.sum()),'사건학습':int(y[tr].sum()),'n평가':int(te.sum()),'사건평가':int(y[te].sum())}
    def ev(p,lab):
        yt=y[te].values; top=np.argsort(-p)[:max(1,int(len(p)*.1))]
        out[lab]=f'AUC {roc_auc_score(yt,p):.3f} / PR {average_precision_score(yt,p):.3f} / 상위10%정밀 {yt[top].mean():.3f}'
    ev(d.브랜드률[te].values+1e-9*np.random.RandomState(0).rand(te.sum()),'기준선②브랜드평균')
    Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    lr=LogisticRegression(C=0.3,max_iter=2000,class_weight='balanced').fit(Z[tr],y[tr]); ev(lr.predict_proba(Z[te])[:,1],'로지스틱')
    gb=HistGradientBoostingClassifier(max_depth=3,learning_rate=0.05,max_iter=300,random_state=0).fit(X[tr],y[tr]); ev(gb.predict_proba(X[te])[:,1],'GBM')
    # 브랜드 내 설명력: 브랜드률 뺀 로지스틱, 브랜드 고정 후 순위
    ev(lr.predict_proba(Z[te])[:,1]-0,'(참고)로지스틱')
    out['상위계수']=dict(sorted(zip(Z.columns,lr.coef_[0]),key=lambda x:-abs(x[1]))[:8])
    print('\n##',name); [print(' ',k_,v) for k_,v in out.items()]
alive1=(s.close.isna())|(s.close>s.open+pd.DateOffset(years=1))
run('A 개업시점→3년내 실제폐점',0,3,'진짜폐점',s.연>0)
run('A2 개업시점→3년내 인허가폐업(재등록 포함)',0,3,None,s.연>0)
run('A3 개업시점→2년내 실제폐점',0,2,'진짜폐점',s.연>0)
run('B 개업1년 생존점→이후 2년내 실제폐점(조기경보)',1,2,'진짜폐점',alive1)

# 같은 브랜드 안에서 입지 변수만으로 구별되는가 (A 기준), 상위10% 정밀도 차이 부트스트랩
def within(t0,h,ycol,cond):
    d=s[cond].copy(); f=feats(t0).loc[d.index]; d=pd.concat([d,f],axis=1)
    tend=d.open+pd.DateOffset(years=t0+h); d=d[tend<=END]
    y=((d.close<=d.open+pd.DateOffset(years=t0+h))&d[ycol]).astype(int)
    tr=d.연<=2021; te=~tr
    loc=['면적','휴게','다중','분기']+list(f.columns)
    X=pd.get_dummies(d[loc+['구','브랜드']],columns=['구','브랜드'],dtype=float); Z=(X-X[tr].mean())/X[tr].std().replace(0,1)
    lr=LogisticRegression(C=0.3,max_iter=2000).fit(Z[tr],y[tr]); p=pd.Series(lr.predict_proba(Z[te])[:,1],index=d.index[te])
    # 브랜드 더미 포함 → 브랜드 효과 통제, 브랜드 내부 순위만 평가
    A=[];W=[]
    for b,g in d[te].groupby('브랜드'):
        yy=y.loc[g.index]
        if 0<yy.sum()<len(yy): A.append(roc_auc_score(yy,p.loc[g.index])); W.append(yy.sum())
    print('\n## 브랜드 내부 AUC(입지·경쟁 변수, 사건 가중 평균)',round(np.average(A,weights=W),3),'브랜드 수',len(A),'사건',sum(W))
    rs=np.random.RandomState(1); yt=y[te].values
    br=y[tr].groupby(d.브랜드[tr]).agg(['sum','count']); pr=y[tr].mean(); base=d.브랜드[te].map((br['sum']+20*pr)/(br['count']+20)).fillna(pr).values
    full=(base*0)+p.values
    diffs=[]
    for _ in range(1000):
        ix=rs.randint(0,len(yt),len(yt)); k=int(len(ix)*.1)
        a_=yt[ix][np.argsort(-(base[ix]+1e-9*rs.rand(len(ix))))[:k]].mean(); b_=yt[ix][np.argsort(-full[ix])[:k]].mean(); diffs.append(b_-a_)
    print('상위10% 정밀도 차(브랜드더미+입지 모델 − 브랜드평균) 95%CI',np.percentile(diffs,[2.5,50,97.5]).round(3))
within(0,3,'진짜폐점',s.연>0)
