# 폐업 보조지표 만들기 + 하나씩 검증 (치킨 B_넓음, 2018–22 개업, y = 3년 안 실제 폐업)
#  모든 지표는 '개업 시점에 알 수 있는 값'만 쓴다(개업 후 정보 금지). 생활인구가 2017–22라 개업 2018–22만 씀.
#  지표 묶음: 브랜드(서울 과거 폐업률·공정위 전년 이탈/확장/명의변경·업력), 같은 브랜드(반경·같은 상권), 경쟁·과열(반경·상권·최근 개폐업),
#            수요(저녁 생활인구·거주비·연령·증가·역), 동 생활 패턴(저녁/낮·20대), 상권 유형, 점포(면적·층·자리 이력·업태·개업월)
#  검증: 지표 하나씩 '브랜드 과거 폐업률 + 면적 + 개업연도'를 통제한 로지스틱에 넣어 OR(1 표준편차당)·p·BH 보정 q,
#        그리고 두 기간(2018–20, 2021–22)에서 방향이 같은지(재현). 결과 → data/processed/indicator_tests.csv, 표본 → _indicators.pkl
import os,io,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from scipy.spatial import cKDTree
from statsmodels.stats.multitest import multipletests
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
def load(name):
    spec=importlib.util.spec_from_file_location(name,f'{O}/code/{name}.py'); m=importlib.util.module_from_spec(spec)
    with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(m)
    return m
cw=load('chicken_wide'); aa=load('area_assign')
c=cw.STORES['B_넓음'].copy(); c['연']=c.open.dt.year
c['y']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
L=pd.read_pickle(f'{P}/location_units_sample.pkl')          # 2018–22 개업, 반경별 생활인구·경쟁·역 값(location_units.py)
d=L[['관리번호','D500','R500','Y500','D2_500','C500','l역','동','상권']].merge(c,on='관리번호',how='left')
d['브랜드']=d.브랜드.fillna('개인'); d['면적']=np.log1p(pd.to_numeric(d['소재지면적'],errors='coerce').fillna(40))
c['브랜드']=c.브랜드.fillna('개인')
o=d.open.values; co=c.open.values; cc=c.close.fillna(pd.Timestamp('2100-01-01')).values; Y1=np.timedelta64(365,'D')

# ---- 브랜드 ----
h=c[c.연.between(2010,2017)]; g=h.groupby('브랜드').y.agg(['sum','size']); k=20
br=(g['sum']+k*h.y.mean())/(g['size']+k); d['브랜드률']=d.브랜드.map(br).fillna(h.y.mean())
d['l브랜드률']=np.log(d.브랜드률/(1-d.브랜드률))
first=c[c.브랜드!='개인'].groupby('브랜드').open.min()
d['브랜드업력']=np.where(d.브랜드=='개인',np.nan,(d.open-d.브랜드.map(first)).dt.days/365)
bt=pd.read_csv(f'{P}/chicken_brand_table.csv').dropna(subset=['ftc']).set_index('브랜드').ftc
s=pd.read_csv(f'{O}/data/raw/ftc_brand_frcs_stats_2017_2025.csv'); s=s[s.indutyMlsfcNm=='치킨']
s=s.groupby(['brandNm','yr'],as_index=False)[['frcsCnt','newFrcsRgsCnt','ctrtEndCnt','ctrtCncltnCnt','nmChgCnt']].sum()
s['이탈률']=(s.ctrtEndCnt+s.ctrtCncltnCnt)/s.frcsCnt; s['확장률']=s.newFrcsRgsCnt/s.frcsCnt; s['명의변경률']=s.nmChgCnt/s.frcsCnt; s['l가맹점']=np.log(s.frcsCnt)
for v in ['이탈률','확장률','명의변경률','l가맹점']:
    S=dict(zip(zip(s.brandNm,s.yr),s[v]))
    d['공정위_'+v]=[S.get((bt.get(b),y-1),np.nan) for b,y in zip(d.브랜드,d.연)]

# ---- 같은 브랜드·경쟁·과열 (치킨집) ----
CT=cKDTree(c[['X','Y']].values); cb=c.브랜드.values
gi_all,_,_=aa.assign(c.X.values,c.Y.values); gi,_,_=aa.assign(d.X.values,d.Y.values); d['상권밖']=(gi<0).astype(int)
def nb(r): return [np.array(q,dtype=int) for q in CT.query_ball_point(d[['X','Y']].values,r)]
q500,q1000=nb(500),nb(1000)
act=lambda q,t:(co[q]<t)&(cc[q]>t)
d['같은브랜드500']=[(act(q,t)&(cb[q]==b)).sum() if b!='개인' else np.nan for q,t,b in zip(q500,o,d.브랜드)]
d['같은브랜드1km']=[(act(q,t)&(cb[q]==b)).sum() if b!='개인' else np.nan for q,t,b in zip(q1000,o,d.브랜드)]
d['신규개업500_1y']=[((co[q]<t)&(co[q]>=t-Y1)).sum() for q,t in zip(q500,o)]
d['폐업500_1y']=[((cc[q]<t)&(cc[q]>=t-Y1)).sum() for q,t in zip(q500,o)]
d['폐업비500_1y']=d.폐업500_1y/(d.C500+d.폐업500_1y+1)
d['프랜비중500']=[(act(q,t)&(cb[q]!='개인')).sum()/max(act(q,t).sum(),1) for q,t in zip(q500,o)]
sc,sb=[],[]
for i,(g_,t,b) in enumerate(zip(gi,o,d.브랜드)):
    if g_<0: sc.append(np.nan); sb.append(np.nan); continue
    m=(gi_all==g_)&(co<t)&(cc>t); sc.append(m.sum()); sb.append((m&(cb==b)).sum() if b!='개인' else np.nan)
d['상권치킨수']=sc; d['상권같은브랜드']=sb

# ---- 전체 음식점 (상권 활성도·치킨 비중·자리 이력) ----
a=cw.a; ax=pd.to_numeric(a['좌표정보(X)'],errors='coerce'); ay=pd.to_numeric(a['좌표정보(Y)'],errors='coerce'); ok=ax.notna().values
AT=cKDTree(np.c_[ax[ok],ay[ok]]); ao=a.open.values[ok]; ac=a.close.fillna(pd.Timestamp('2100-01-01')).values[ok]
qa=[np.array(q,dtype=int) for q in AT.query_ball_point(d[['X','Y']].values,500)]
d['음식점500']=[((ao[q]<t)&(ac[q]>t)).sum() for q,t in zip(qa,o)]
d['음식점폐업비500_1y']=[((ac[q]<t)&(ac[q]>=t-Y1)).sum()/max(((ao[q]<t)&(ac[q]>t)).sum(),1) for q,t in zip(qa,o)]
d['치킨비중500']=d.C500/(d.음식점500+1)
qs=[np.array(q,dtype=int) for q in AT.query_ball_point(d[['X','Y']].values,10)]
d['자리폐업10y']=[((ac[q]<t)&(ac[q]>=t-np.timedelta64(3650,'D'))).sum() for q,t in zip(qs,o)]

# ---- 수요·생활인구 ----
d['l저녁인구500']=np.log(d.D500); d['거주비']=d.R500/d.D500; d['비중3049']=d.Y500/d.D500
d['저녁인구증가']=np.log(d.D500/d.D2_500); d['l수요경쟁비']=np.log(d.D500/(d.C500+1)); d['l경쟁500']=np.log1p(d.C500)
lp=pd.read_csv(f'{P}/living_pop_dong_year.csv',dtype={'동':str}); lp['저녁낮비']=lp.저녁/lp.낮; lp['밤저녁비']=lp.밤/lp.저녁; lp['비중2029']=lp.저녁2029/lp.저녁
LPm=lp.set_index(['동','연'])
for v in ['저녁낮비','밤저녁비','비중2029']: d['동_'+v]=[LPm[v].get((str(x),y-1),np.nan) for x,y in zip(d.동,d.연)]
T=pd.read_csv(f'{P}/area_types_people.csv').set_index('TRDAR_CD').유형; code=pd.to_numeric(aa.A.상권코드).values
d['사람유형']=[{0:'직장가',1:'주거지',2:'먹자골목'}.get(T.get(code[i]),'상권밖') if i>=0 else '상권밖' for i in gi]
d['공식유형']=np.where(gi>=0,aa.A.상권유형.values[np.clip(gi,0,None)],'상권밖')

# ---- 점포 ----
addr=d.지번주소.fillna(''); d['지하']=addr.str.contains('지하|B1').astype(int); d['2층이상']=addr.str.contains(r'[2-9]층|1\d층').astype(int)
d['휴게']=(d.src=='cafe').astype(int); d['프랜차이즈']=(d.브랜드!='개인').astype(int); d['여름개업']=d.open.dt.month.between(5,8).astype(int)
d['다중이용']=(d.다중이용업소여부=='Y').astype(int)

IND={  # 이름: (컬럼, 묶음, 대상)
 '브랜드 서울 과거 폐업률(2010–17)':('l브랜드률','브랜드','전체'),'프랜차이즈 여부':('프랜차이즈','브랜드','전체'),
 '브랜드 서울 업력(년)':('브랜드업력','브랜드','프랜'),'공정위 전년 이탈률':('공정위_이탈률','브랜드','프랜'),'공정위 전년 확장률(신규/가맹점)':('공정위_확장률','브랜드','프랜'),
 '공정위 전년 명의변경률':('공정위_명의변경률','브랜드','프랜'),'공정위 전년 가맹점 수(log)':('공정위_l가맹점','브랜드','프랜'),
 '같은 브랜드 500m':('같은브랜드500','같은 브랜드','프랜'),'같은 브랜드 1km':('같은브랜드1km','같은 브랜드','프랜'),'같은 상권 같은 브랜드':('상권같은브랜드','같은 브랜드','프랜'),
 '500m 치킨집(log)':('l경쟁500','경쟁·과열','전체'),'같은 상권 치킨집 수':('상권치킨수','경쟁·과열','전체'),'최근 1년 500m 치킨 신규 개업':('신규개업500_1y','경쟁·과열','전체'),
 '최근 1년 500m 치킨 폐업':('폐업500_1y','경쟁·과열','전체'),'최근 1년 500m 치킨 폐업 비율':('폐업비500_1y','경쟁·과열','전체'),'500m 치킨집 중 프랜차이즈 비중':('프랜비중500','경쟁·과열','전체'),
 '500m 음식점 수':('음식점500','상권 활성','전체'),'최근 1년 500m 음식점 폐업 비율':('음식점폐업비500_1y','상권 활성','전체'),'500m 음식점 중 치킨 비중':('치킨비중500','상권 활성','전체'),
 '500m 저녁 생활인구(log)':('l저녁인구500','수요·생활인구','전체'),'거주비(새벽/저녁)':('거주비','수요·생활인구','전체'),'30–40대 비중(저녁)':('비중3049','수요·생활인구','전체'),
 '저녁 인구 1년 증가율':('저녁인구증가','수요·생활인구','전체'),'저녁인구/치킨집(log)':('l수요경쟁비','수요·생활인구','전체'),'역 저녁 하차(log)':('l역','수요·생활인구','전체'),
 '동 저녁/낮 인구비':('동_저녁낮비','수요·생활인구','전체'),'동 밤/저녁 인구비':('동_밤저녁비','수요·생활인구','전체'),'동 20대 비중(저녁)':('동_비중2029','수요·생활인구','전체'),
 '상권 밖 입지':('상권밖','상권 유형','전체'),'면적(log)':('면적','점포','전체'),'지하':('지하','점포','전체'),'2층 이상':('2층이상','점포','전체'),
 '휴게음식점 등록':('휴게','점포','전체'),'자리 이력(10m, 10년 음식점 폐업)':('자리폐업10y','점포','전체'),'여름(5–8월) 개업':('여름개업','점포','전체'),'다중이용업소':('다중이용','점포','전체'),
}
if __name__=='__main__':
    print(f'표본 2018–22 개업 {len(d)}곳 (프랜차이즈 {d.프랜차이즈.sum()}), 3년 실제 폐업률 {d.y.mean():.3f}')
    rows=[]
    for nm,(col,grp,tg) in IND.items():
        x=d if tg=='전체' else d[d.프랜차이즈==1]; x=x.dropna(subset=[col]).copy()
        if x[col].std()==0: continue
        x['z']=(x[col]-x[col].mean())/x[col].std()
        ctrl='l브랜드률+면적+C(연)' if col not in('l브랜드률','면적','프랜차이즈') else 'C(연)'+('+면적' if col!='면적' else '')+('+l브랜드률' if col=='면적' else '')
        m=smf.logit(f'y~{ctrl}+z',x).fit(disp=0)
        p1=smf.logit(f'y~{ctrl}+z',x[x.연<=2020]).fit(disp=0).params['z']; p2=smf.logit(f'y~{ctrl}+z',x[x.연>=2021]).fit(disp=0).params['z']
        hi=x[col]>=x[col].quantile(.8) if x[col].nunique()>5 else x[col]>x[col].min(); lo=x[col]<=x[col].quantile(.2) if x[col].nunique()>5 else x[col]==x[col].min()
        rows.append(dict(지표=nm,묶음=grp,대상=tg,n=len(x),하위_폐업률=x[lo].y.mean(),상위_폐업률=x[hi].y.mean(),OR_1SD=np.exp(m.params['z']),p=m.pvalues['z'],
                         재현=('같음' if np.sign(p1)==np.sign(p2) else '뒤집힘'),OR_18_20=np.exp(p1),OR_21_22=np.exp(p2)))
    r=pd.DataFrame(rows); r['q_BH']=multipletests(r.p,method='fdr_bh')[1]
    r['판정']=np.where((r.q_BH<.05)&(r.재현=='같음'),'채택',np.where((r.p<.1)&(r.재현=='같음'),'약함','기각'))
    pd.set_option('display.width',300); pd.set_option('display.max_colwidth',30)
    print('\n[지표별 검증] 통제: 브랜드 과거 폐업률+면적+개업연도. 하위/상위 = 하위·상위 20%(이진은 0 vs 1) 폐업률')
    print(r.sort_values('p').round(3).to_string(index=False))
    r.to_csv(f'{P}/indicator_tests.csv',index=False,encoding='utf-8-sig')
    # 사용자 가설 표
    f=d[d.프랜차이즈==1]
    print('\n[가설1] 같은 브랜드 수 구간별 3년 실제 폐업률 (프랜차이즈)')
    for col in ['같은브랜드500','같은브랜드1km','상권같은브랜드']:
        t=f.dropna(subset=[col]).assign(g=lambda x:pd.cut(x[col],[-1,0,1,2,99],labels=['0','1','2','3+'])).groupby('g',observed=True).y.agg(['mean','size'])
        print(f'  {col}: '+' | '.join(f'{i}: {r_['mean']:.3f}(n={int(r_['size'])})' for i,r_ in t.iterrows()))
    print('\n[가설2] 생활인구 패턴 5분위별 3년 실제 폐업률 (전체)')
    for col in ['거주비','비중3049','저녁인구증가','l저녁인구500','동_저녁낮비','동_밤저녁비','동_비중2029','l수요경쟁비']:
        t=d.dropna(subset=[col]).assign(g=lambda x:pd.qcut(x[col],5,labels=False,duplicates='drop')).groupby('g').y.mean()
        print(f'  {col}: '+' → '.join(f'{v:.3f}' for v in t.values))
    print('\n[가설3] 사람유형·공식유형별 폐업률'); print('  ',d.groupby('사람유형').y.agg(['mean','size']).round(3).to_dict('index')); print('  ',d.groupby('공식유형').y.agg(['mean','size']).round(3).to_dict('index'))
    d.drop(columns=[x for x in d.columns if x in('지번주소','사업장명','nm','도로명주소')]).to_pickle(f'{P}/_indicators.pkl')
