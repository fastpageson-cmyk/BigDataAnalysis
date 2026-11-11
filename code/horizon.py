# 전 기간 생존: 개업 후 연차별 폐업 위험률(hazard)과 누적 폐업률, 프랜차이즈 vs 전체 음식점
import os,numpy as np,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); END=pd.Timestamp('2026-11-04')
def table(d,flag,maxy=15):
    age_end=((d.close.where(flag,pd.NaT).fillna(END).clip(upper=END))-d.open).dt.days/365.25
    ev=flag&(d.close<=END)
    # 연차별 위험률: 그 연차를 시작한 점포 중 그 해에 폐업한 비율 (중도절단 반영한 생명표)
    R=[];S=1
    for k in range(maxy):
        risk=(age_end>k).sum(); e=((age_end>k)&(age_end<=k+1)&ev).sum(); cens=((age_end>k)&(age_end<=k+1)&~ev).sum()
        h=e/(risk-cens/2) if risk else np.nan; S*=1-h; R.append(dict(연차=k+1,위험점포=risk,폐업=e,위험률=round(h,4),누적폐업=round(1-S,3)))
    return pd.DataFrame(R)
s=pd.read_pickle(f'{O}/data/processed/_brand_stores_y.pkl'); s=s[(s.n_match==1)&(s.브랜드!='스타벅스(직영)')&s.open.dt.year.between(2005,2025)]
print('## 프랜차이즈 실제폐점(재등록·이전 제외)'); print(table(s,s.진짜폐점).to_string(index=False))
print('## 프랜차이즈 인허가폐업 전체'); print(table(s,s.close.notna()).to_string(index=False))
a=pd.read_pickle(f'{O}/data/processed/_lic_all.pkl'); a['open']=pd.to_datetime(a['인허가일자'].str.strip(),errors='coerce'); a['close']=pd.to_datetime(a['폐업일자'].str.strip(),errors='coerce')
a=a[a.open.dt.year.between(2005,2025)&~a.관리번호.isin(s.관리번호)]
print('## 비프랜차이즈 포함 전체 음식점 인허가폐업(참고)'); print(table(a,a.close.notna()).to_string(index=False))
# 업종별 실제폐점 누적
for g,d in s.groupby('업종'):
    t=table(d,d.진짜폐점,8); print(g,len(d),'누적폐점 1/2/3/5/8년',t.누적폐업.iloc[[0,1,2,4,7]].tolist(),'위험률 최대 연차',int(t.연차[t.위험률.idxmax()]))
