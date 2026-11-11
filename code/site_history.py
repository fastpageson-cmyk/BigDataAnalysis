# 자리 이력: "이 점포가 망할까" 대신 "이 자리는 가게가 자주 바뀌는 자리인가"를 본다
#  [1] 전 업종 음식점(인허가 전체): 같은 자리(반경 R)의 직전 가게 수명·최근 5년 단명(2년 미만) 횟수 → 새 가게 3년 폐업. R=5m(같은 건물 좌표)·30m 비교, 기간 반반
#  [2] 치킨집(chicken_wide.py B_넓음, 3년 실제 폐업): 같은 지표, 프랜차이즈/개인, 기간 반반, 통제 추가
#  [3] 거름망: 1차 브랜드 → 2차 자리 이력 → 3차 경쟁 → 4차 상권 유형 순으로 걸러 남은 점포의 폐업률이 단계마다 내려가나
#      기준은 전부 2010–18 개업에서 정하고 2019–22 개업에 적용. 같은 수만큼 모형(HGB)이 고른 점포와 비교
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.metrics import roc_auc_score
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04'); YR=np.timedelta64(365,'D')
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
bf=load('brand_fit'); cw=bf.cw; a=cw.a
al=pd.DataFrame({'X':pd.to_numeric(a['좌표정보(X)'],errors='coerce'),'Y':pd.to_numeric(a['좌표정보(Y)'],errors='coerce'),'o':a.open,'c':a.close}).dropna(subset=['X','o']).reset_index(drop=True)
ao=al.o.values; ac=al.c.fillna(pd.Timestamp('2100-01-01')).values; AXY=al[['X','Y']].values; AT=cKDTree(AXY)

def site(xy,t,R,skip=None):
    """시점 t 이전에 반경 R 안에서 문 닫은 음식점들로 만든 자리 이력"""
    out=[]
    for k,(q,s) in enumerate(zip(AT.query_ball_point(xy,R),t)):
        q=np.array(q,int); q=q[ac[q]<s]
        if not len(q): out.append((0,np.nan,0,np.nan)); continue
        ten=(ac[q]-ao[q])/YR; last=ac[q].argmax(); rec=ac[q]>=s-5*YR
        out.append((len(q),ten[last],int((rec&(ten<2)).sum()),(s-ac[q][last])/YR))
    return pd.DataFrame(out,columns=['이전수','직전수명','단명5y','공실년'])
def tab(d,y,col,bins,labels):
    g=d.groupby(pd.cut(d[col],bins,labels=labels,right=False))[y].agg(['mean','size']); return ', '.join(f'{k} {r["mean"]:.3f}({int(r["size"])})' for k,r in g.iterrows())

if __name__=='__main__':
    # ---------- 1. 전 업종 ----------
    m=(al.o>=pd.Timestamp('2010-01-01'))&(al.o<=END-pd.DateOffset(years=3)); r=al[m].copy(); r['y3']=(r.c<=r.o+pd.DateOffset(years=3)).astype(int); r['연']=r.o.dt.year
    print(f'[1] 전 업종 음식점 2010–2023 개업 {len(r)}곳, 3년 신고 폐업률 {r.y3.mean():.3f}')
    for R in (5,30):
        s=site(r[['X','Y']].values,r.o.values,R); s.index=r.index; d=r.join(s); h=d[d.이전수>0]
        print(f'-- 같은 자리 반경 {R}m: 이전 가게가 있던 자리 {len(h)/len(d):.2f}, 없던 자리 폐업률 {d[d.이전수==0].y3.mean():.3f}')
        print('   직전 가게 수명별:',tab(h,'y3','직전수명',[0,1,2,3,6,100],['1년 미만','1–2년','2–3년','3–6년','6년 이상']))
        print('   최근 5년 단명(2년 미만) 횟수별:',tab(h,'y3','단명5y',[0,1,2,3,100],['0회','1회','2회','3회 이상']))
        for nm,g in [('2010–15',h[h.연<=2015]),('2016–23',h[h.연>=2016])]: print(f'   {nm} 직전 1년 미만 vs 6년 이상: {g[g.직전수명<1].y3.mean():.3f} vs {g[g.직전수명>=6].y3.mean():.3f}')
        mm=smf.logit('y3~C(연)+np.log(직전수명.clip(lower=0.1))+단명5y',h).fit(disp=0); print(f'   로지스틱(연도 통제): 직전수명 2.7배마다 OR {np.exp(mm.params.iloc[-2]):.2f}, 단명 1회마다 OR {np.exp(mm.params.iloc[-1]):.2f}, 단독 AUC {roc_auc_score(h.y3,mm.predict(h)):.3f}')

    # ---------- 2. 치킨 ----------
    c=bf.c.reset_index(drop=True).copy(); c['y3']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
    co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values; CT=cKDTree(c[['X','Y']].values)
    d=c[c.연.between(2010,2022)].copy(); s=site(d[['X','Y']].values,d.open.values,30); s.index=d.index; d=d.join(s)
    d['치킨500']=[((co[q]<o)&(cc[q]>o)).sum() for q,o in zip(CT.query_ball_point(d[['X','Y']].values,500),d.open.values)]
    d['프랜']=d.프랜차이즈.astype(int); d['회전자리']=((d.직전수명<1)|(d.단명5y>=2)).astype(int)
    print(f'\n[2] 치킨집 2010–22 개업 {len(d)}곳(3년 실제 폐업). 회전 잦은 자리 = 직전 가게 1년 미만 또는 최근 5년 단명 2회 이상 ({d.회전자리.mean():.2f})')
    for nm,g in [('전체',d),('프랜차이즈',d[d.프랜==1]),('개인',d[d.프랜==0]),('2010–15 개업',d[d.연<=2015]),('2016–22 개업',d[d.연>=2016])]:
        f='y3~면적+C(연)+np.log1p(치킨500)+회전자리'+('+프랜' if g.프랜.nunique()>1 else '')
        mm=smf.logit(f,g).fit(disp=0); print(f'  {nm} n={len(g)}: 회전 잦은 자리 {g[g.회전자리==1].y3.mean():.3f} vs 그 외 {g[g.회전자리==0].y3.mean():.3f}, OR {np.exp(mm.params["회전자리"]):.2f} (p={mm.pvalues["회전자리"]:.4f})')
    print('  단명 횟수별(전체):',tab(d,'y3','단명5y',[0,1,2,3,100],['0회','1회','2회','3회 이상']))

    # ---------- 3. 거름망 ----------
    tr=d[d.연<=2018]; te=d[d.연>=2019].copy()
    g=tr[tr.프랜==1].groupby('브랜드').y3.agg(['sum','size']); p0=tr[tr.프랜==1].y3.mean(); enc=(g['sum']+20*p0)/(g['size']+20)
    good=set(enc[enc<=enc.median()].index); q50=tr.치킨500.median()
    te['S1']=te.브랜드.isin(good); te['S2']=te.회전자리==0; te['S3']=te.치킨500<=q50; te['S4']=~te.사람유형.isin(['직장가','먹자골목'])
    NM={'S1':'1차 브랜드(학습 구간 폐업률 하위 절반 프랜차이즈)','S2':'2차 자리 이력(회전 잦은 자리 제외)','S3':f'3차 경쟁(500m 치킨집 {q50:.0f}곳 이하)','S4':'4차 상권 유형(직장가·먹자골목 제외)'}
    print(f'\n[3] 거름망: 기준은 2010–18 개업에서 정함 → 2019–22 개업 {len(te)}곳에 적용 (전체 3년 실제 폐업률 {te.y3.mean():.3f})')
    keep=np.ones(len(te),bool)
    for k in ['S1','S2','S3','S4']:
        prev=keep.copy(); keep=keep&te[k].values; out=prev&~te[k].values
        print(f'  {NM[k]}: 통과 {keep.sum()}곳 폐업률 {te.y3[keep].mean():.3f} | 이 단계에서 걸러진 {out.sum()}곳 {te.y3[out].mean():.3f}')
    fr=te[te.프랜==1]; keep=np.ones(len(fr),bool); print(f'  -- 프랜차이즈 {len(fr)}곳만, 브랜드는 그대로 두고 입지 거름망(2~4차)만: 시작 폐업률 {fr.y3.mean():.3f}')
    for k in ['S2','S3','S4']:
        prev=keep.copy(); keep=keep&fr[k].values; out=prev&~fr[k].values; print(f'     {NM[k][:2]} 통과 {keep.sum()}곳 {fr.y3[keep].mean():.3f} | 걸러진 {out.sum()}곳 {fr.y3[out].mean():.3f}')
    fr2=fr.assign(통과=keep.astype(int)); mm=smf.logit('y3~C(브랜드)+면적+통과',fr2[fr2.브랜드.map(fr2.브랜드.value_counts())>=10]).fit(disp=0,method='bfgs',maxiter=2000)
    print(f'     입지 거름망 통과 효과(브랜드·면적 통제): OR {np.exp(mm.params["통과"]):.2f}, p={mm.pvalues["통과"]:.3f}')
    # 같은 수만큼 모형이 고른 점포와 비교
    tr2=tr.copy(); te['브랜드enc']=np.where(te.프랜==1,te.브랜드.map(enc).fillna(p0),tr[tr.프랜==0].y3.mean()); tr2['브랜드enc']=np.where(tr2.프랜==1,tr2.브랜드.map(enc).fillna(p0),tr[tr.프랜==0].y3.mean())
    cols=['프랜','브랜드enc','면적','치킨500','직전수명','단명5y','이전수']; te['유형c']=te.사람유형.astype('category').cat.codes; tr2['유형c']=tr2.사람유형.astype('category').cat.codes
    for nm,cs in [('브랜드+면적',['프랜','브랜드enc','면적']),('브랜드+면적+거름망에 쓴 입지 변수',cols+['유형c'])]:
        p=HGB(max_depth=3,learning_rate=0.05,max_iter=200,min_samples_leaf=40,random_state=0).fit(tr2[cs],tr2.y3).predict_proba(te[cs])[:,1]
        full=te.S1&te.S2&te.S3&te.S4; k=int(full.sum()); top=np.argsort(p)[:k]
        print(f'  모형({nm}) AUC {roc_auc_score(te.y3,p):.3f}: 가장 안전하다고 본 {k}곳 폐업률 {te.y3.values[top].mean():.3f} (거름망 4단계 통과 {k}곳은 {te.y3[full].mean():.3f})')
    score=te[['S1','S2','S3','S4']].sum(1); print(f'  거름망 통과 개수를 점수로 쓴 AUC {roc_auc_score(te.y3,-score):.3f}, 통과 개수별 폐업률:',te.groupby(score).y3.agg(['mean','size']).round(3).to_dict('index'))
