# 서울시 상권분석서비스(행정동, 점포·추정매출 2021Q1–2025Q4)로 치킨전문점(CS100007) 시장 보기
#  1) 현황: 연도별 점포 수·프랜차이즈 비중·개폐업·매출·점포당 매출·시간대/연령/요일 구성
#  2) 교차검증: 서비스 폐업 수 vs 인허가 치킨집(chicken_wide.py B_넓음) 폐업 수 (연도, 동)
#  3) 동 패널: 올해 동 상태(점포당 매출·점포 수·변화) → 다음 해 치킨 폐업률 / 점포가 늘면 점포당 매출이 주나(포화)
#  4) 점포 단위: 2022-01~2023-11 개업 치킨집 3년 실제 폐업 ~ 개업 전년 동 점포당 매출 (브랜드·프랜차이즈 통제)
import os,io,glob,json,contextlib,importlib.util,numpy as np,pandas as pd,warnings; warnings.filterwarnings('ignore')
import statsmodels.formula.api as smf
from shapely.geometry import shape
from shapely import STRtree,points
from shapely.ops import transform as stf
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); R=f'{O}/data/raw/상권분석_행정동'
rd=lambda k:pd.concat([pd.read_csv(f,encoding='cp949') for f in sorted(glob.glob(f'{R}/{k}/*.csv'))])
st=rd('점포'); sl=rd('추정매출'); CK='CS100007'
st=st[st.서비스_업종_코드==CK]; sl=sl[sl.서비스_업종_코드==CK]
for x in (st,sl): x['연']=x.기준_년분기_코드//10; x['동']=x.행정동_코드.astype(str)
st['전체']=st.유사_업종_점포_수

# 1) 현황
a=st.groupby('연').agg(점포_분기평균=('전체','sum'),프랜차이즈=('프랜차이즈_점포_수','sum'),개업=('개업_점포_수','sum'),폐업=('폐업_점포_수','sum'))
a[['점포_분기평균','프랜차이즈']]/=4; a['프랜차이즈비중']=a.프랜차이즈/a.점포_분기평균
s=sl.groupby('연')[[c for c in sl.columns if c.endswith('매출_금액')]].sum()
a['매출_억']=s.당월_매출_금액/1e8; a['점포당매출_만원']=s.당월_매출_금액/a.점포_분기평균/1e4
a['저녁밤17_24']=(s['시간대_17~21_매출_금액']+s['시간대_21~24_매출_금액'])/s.당월_매출_금액
a['주말']=s.주말_매출_금액/s.당월_매출_금액
for g in ['20','30','40','50','60_이상']: a[f'연령{g}']=s[f'연령대_{g}_매출_금액']/s.당월_매출_금액
print('[1] 서울 치킨전문점 연도별 현황 (점포는 분기 평균, 매출은 카드 기반 추정)'); print(a.round(3).to_string())

# 2) 교차검증: 인허가 치킨집 폐업(재등록 포함 신고 기준) 연·동별
def load(n):
    sp=importlib.util.spec_from_file_location(n,f'{O}/code/{n}.py'); m=importlib.util.module_from_spec(sp)
    with contextlib.redirect_stdout(io.StringIO()): sp.loader.exec_module(m)
    return m
cw=load('chicken_wide'); c=cw.STORES['B_넓음'].copy()
tf=Transformer.from_crs(5179,5174,always_xy=True).transform
g=json.load(open(f'{O}/data/raw/행정동경계/HangJeongDong_ver20200701.geojson')); F=[f for f in g['features'] if f['properties']['sido']=='11']
DG=[stf(tf,shape(f['geometry'])) for f in F]; DC=np.array([f['properties']['adm_cd2'][:8] for f in F]); DT=STRtree(DG)
i=DT.query(points(c.X.values,c.Y.values),predicate='within'); c['동']=None; c.iloc[i[0],c.columns.get_loc('동')]=DC[i[1]]
print(f'\n[2] 교차검증: 인허가 치킨집 행정동 배정률 {c.동.notna().mean():.3f}, 서비스 동코드 중 2020 경계와 일치 {st.동.drop_duplicates().isin(DC).mean():.3f}')
c['폐업연']=c.close.dt.year; li=c[c.폐업연.between(2021,2025)]
cmp=pd.DataFrame({'서비스_폐업':a.폐업,'인허가_폐업':li.groupby('폐업연').size(),'인허가_실제폐업(재등록 제외)':li[~li.재등록].groupby('폐업연').size()}); print(cmp.to_string())
dd=st.groupby(['동','연']).폐업_점포_수.sum().rename('서비스').to_frame().join(li.groupby(['동','폐업연']).size().rename_axis(['동','연']).rename('인허가'),how='left').fillna(0)
print(f'  동×연 폐업 수 상관: {dd.corr().iloc[0,1]:.2f} (동 {dd.index.get_level_values(0).nunique()}개 × 5년)')

# 3) 동 패널 (연 단위)
p=st.groupby(['동','연']).agg(점포=('전체','mean'),폐업=('폐업_점포_수','sum'),개업=('개업_점포_수','sum'),프랜=('프랜차이즈_점포_수','mean'))
p=p.join(sl.groupby(['동','연'])[['당월_매출_금액','시간대_17~21_매출_금액','시간대_21~24_매출_금액']].sum())
p=p[p.점포>=3].copy(); p['점포당매출']=p.당월_매출_금액/p.점포/1e4; p['l매출']=np.log(p.점포당매출.clip(lower=1)); p['l점포']=np.log(p.점포)
p['프랜비중']=p.프랜/p.점포; p['저녁비중']=(p['시간대_17~21_매출_금액']+p['시간대_21~24_매출_금액'])/p.당월_매출_금액.clip(lower=1)
p=p.reset_index().sort_values(['동','연']); n=p.groupby('동')
p['다음해폐업률']=n.폐업.shift(-1)/p.점포; p['다음해순증']=n.점포.shift(-1)/p.점포-1
p['d매출']=p.l매출-n.l매출.shift(1); p['d점포']=p.l점포-n.l점포.shift(1)
q=p.dropna(subset=['다음해폐업률']); q=q[q.점포당매출>0]
print(f'\n[3] 동 패널: 치킨 점포 3곳 이상 동-연 {len(q)}개(2021–24 → 다음 해). 가중 OLS(점포 수), 연도 고정효과')
def wfit(fml,data):
    data=data.dropna(subset=[v for v in fml.split('~')[1].split('+') if not v.startswith('C(')])
    return smf.wls(fml,data,weights=data.점포).fit(cov_type='cluster',cov_kwds={'groups':data.동.astype('category').cat.codes})
for fml in ['다음해폐업률~l매출+l점포+프랜비중+저녁비중+C(연)','다음해폐업률~l매출+d매출+l점포+프랜비중+C(연)']:
    m=wfit(fml,q); print('  '+fml.split('~')[1]+f' (n={int(m.nobs)}): '+', '.join(f'{k} {m.params[k]:+.3f}(p={m.pvalues[k]:.3f})' for k in m.params.index if not k.startswith('C(') and k!='Intercept'))
q['매출4분위']=pd.qcut(q.l매출,4,labels=['하','중하','중상','상']); print('  점포당 매출 4분위별 다음해 폐업률(가중):',q.groupby('매출4분위').apply(lambda x:np.average(x.다음해폐업률,weights=x.점포)).round(3).to_dict())
r=p.dropna(subset=['d매출','d점포']); m=wfit('d매출~d점포+C(연)',r)
print(f'  포화: 점포 수가 1% 늘면 점포당 매출 {m.params["d점포"]:+.2f}% (p={m.pvalues["d점포"]:.3f}, 동-연 {len(r)}개)')

# 4) 점포 단위: 개업 전년 동 상태 → 3년 실제 폐업
c['연']=c.open.dt.year; c['y3']=((c.close<=c.open+pd.DateOffset(years=3))&~c.재등록).astype(int)
z=c[(c.open>=pd.Timestamp('2022-01-01'))&(c.open<=pd.Timestamp('2023-11-04'))].copy(); z['기준연']=z.연-1
z=z.merge(p[['동','연','l매출','l점포','프랜비중']].rename(columns={'연':'기준연'}),on=['동','기준연'],how='inner')
z['브랜드2']=np.where(z.프랜차이즈,z.브랜드.fillna('기타프랜'),'개인')
print(f'\n[4] 점포 단위: 2022-01~2023-11 개업 치킨집 중 동 매칭 {len(z)}곳(프랜차이즈 {z.프랜차이즈.sum()}), 3년 실제 폐업률 {z.y3.mean():.3f}')
z['매출4분위']=pd.qcut(z.l매출,4,labels=['하','중하','중상','상']); print('  개업 전년 동 점포당 매출 4분위별 3년 폐업률:',z.groupby('매출4분위').y3.agg(['mean','size']).round(3).to_dict('index'))
for lab,fml in [('프랜차이즈 여부 통제','y3~프랜차이즈+l매출+l점포'),('+프랜차이즈 비중','y3~프랜차이즈+l매출+l점포+프랜비중')]:
    m=smf.logit(fml,z).fit(disp=0); print(f'  {lab}: '+', '.join(f'{k} OR {np.exp(m.params[k]):.2f}(p={m.pvalues[k]:.3f})' for k in m.params.index if k!='Intercept'))
for nm,g in [('개인',z[~z.프랜차이즈]),('프랜차이즈',z[z.프랜차이즈])]:
    m=smf.logit('y3~l매출+l점포',g).fit(disp=0); print(f'  {nm}만 n={len(g)}: 점포당매출 OR {np.exp(m.params["l매출"]):.2f}(p={m.pvalues["l매출"]:.3f}), 점포수 OR {np.exp(m.params["l점포"]):.2f}(p={m.pvalues["l점포"]:.3f})')
p.drop(columns=['당월_매출_금액']).round(4).to_csv(f'{O}/data/processed/dong_chicken_panel.csv',index=False,encoding='utf-8-sig')
