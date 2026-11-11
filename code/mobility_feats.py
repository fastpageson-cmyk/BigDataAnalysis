# 이동 데이터 피처: (1) 서울 생활인구(OA-14991) 행정동×연도×시간대 평균, (2) 지하철 역별 시간대 하차(OA-12252) 역×연도 평균
# (3) 점포 좌표 → 행정동(상가정보 최근접 5곳 다수결). 출력은 data/processed/의 csv 3개.
import os,glob,zipfile,io,re,numpy as np,pandas as pd
from scipy.spatial import cKDTree
from pyproj import Transformer
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
tf=Transformer.from_crs(4326,5174,always_xy=True)   # 인허가 좌표계(EPSG:5174)에 맞춤

# 1) 생활인구: 월 파일마다 행정동×시간대 평균 → 연도 평균. 시간대 띠: 낮 11–13시, 저녁 17–20시, 밤 21–23시
BAND={**{h:'낮' for h in (11,12,13)},**{h:'저녁' for h in (17,18,19,20)},**{h:'밤' for h in (21,22,23)}}
out=[]
for z in sorted(glob.glob(f'{O}/data/raw/생활인구/LOCAL_PEOPLE_DONG_*.zip')):
    with zipfile.ZipFile(z) as Z:
        for n in Z.namelist():
            raw=Z.read(n); enc='utf-8-sig' if raw[:3]==b'\xef\xbb\xbf' else 'cp949'   # 월마다 인코딩이 다름
            df=pd.read_csv(io.BytesIO(raw),encoding=enc,usecols=[0,1,2,3],header=0,names=['일','시','동','인구'],dtype={'일':str,'시':int,'동':str})
            df['띠']=df.시.map(BAND); df=df.dropna(subset=['띠'])
            g=df.groupby(['동','띠']).인구.mean().unstack(); g['월']=n[-10:-4]; out.append(g.reset_index())
            print(n,len(df),flush=True)
lp=pd.concat(out); lp['연']=lp.월.str[:4].astype(int)
lp=lp.groupby(['동','연'])[['낮','저녁','밤']].mean().reset_index(); lp.to_csv(f'{P}/living_pop_dong_year.csv',index=False,encoding='utf-8-sig')
print('생활인구',lp.shape,lp.연.min(),lp.연.max(),lp.동.nunique(),'동')

# 2) 지하철: 역명 정리(괄호 부기 제거, 끝 '역' 제거) → 역×연도 월평균 하차(전체, 저녁 17–20시). 환승역은 호선 합산
s=pd.read_csv(f'{O}/data/raw/지하철/subway_time_2015_2026.csv')
norm=lambda x:re.sub(r'\(.*?\)','',str(x)).strip().removesuffix('역') if str(x)!='서울역' else '서울'
s['역']=s.STTN.map(norm).replace({'당고개':'불암산','자양':'뚝섬유원지'}); s['연']=s.USE_MM//100   # 개명역은 현재 이름으로
s['하차']=s.filter(regex=r'^HR_\d+_GET_OFF_NOPE$').sum(axis=1); s['저녁하차']=s[[f'HR_{h}_GET_OFF_NOPE' for h in (17,18,19,20)]].sum(axis=1)
sm=s.groupby(['역','연','USE_MM'])[['하차','저녁하차']].sum().groupby(['역','연']).mean().reset_index()
m=pd.read_csv(f'{O}/data/raw/출입구/seoul_station_master_20260902.csv',encoding='cp949'); m['역']=m.역사명.map(norm)
m=m.groupby('역')[['위도','경도']].mean().reset_index(); m['X'],m['Y']=tf.transform(m.경도.values,m.위도.values)
sm=sm.merge(m[['역','X','Y']],on='역',how='left')
print('지하철 역',sm.역.nunique(),'좌표 매칭 실패',sm[sm.X.isna()].역.nunique(),sorted(sm[sm.X.isna()].역.unique())[:15])
sm.dropna(subset=['X']).to_csv(f'{P}/subway_station_year.csv',index=False,encoding='utf-8-sig')

# 3) 상가정보(2026-06) 좌표·행정동코드 → KD트리. 점포 행정동 배정은 모델 스크립트에서 이 파일로 함
b=pd.read_csv(f'{O}/data/raw/막시무스/상가정보/sbiz_store_info_seoul_202606.csv',usecols=['행정동코드','경도','위도'],dtype={'행정동코드':str}).dropna()
b['X'],b['Y']=tf.transform(b.경도.values,b.위도.values); b['동']=b.행정동코드.str[:8]
b[['동','X','Y']].to_csv(f'{P}/sbiz_dong_points.csv',index=False)
print('상가정보 점',len(b),'동',b.동.nunique(),'생활인구 동과 겹침',len(set(b.동)&set(lp.동)))
