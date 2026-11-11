# 위험 반영 수익: 브랜드별 '창업비 1원당 5년 기대 매출'.
#  연 폐업 위험 h = 1-(1-서울 3년 실제 폐업률)^(1/3) (3년 안에서 위험이 평평하다고 가정, 프랜차이즈는 survival_threshold.py에서 2년차부터 평평)
#  5년 기대 영업연수 = Σ_{t=0..4} (1-h)^t   (그해 초에 살아 있으면 1년치 매출)
#  5년 기대 매출 = 공정위 2024 전국 평균매출 × 기대 영업연수.  배수 = 5년 기대 매출 / 창업비(임차보증금·권리금 제외)
#  폐업률 95% 구간(brand_risk_return.csv lo·hi)으로 같은 계산을 해 범위를 붙인다. 매출은 이익이 아님에 주의.
import os,numpy as np,pandas as pd
O=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); P=f'{O}/data/processed'
b=pd.read_csv(f'{P}/brand_risk_return.csv')
def yrs(p3): h=1-(1-p3)**(1/3); return sum((1-h)**t for t in range(5))
for k,c in [('기대연수','폐업'),('기대연수_비관','hi'),('기대연수_낙관','lo')]: b[k]=b[c].div(100).map(yrs)
b['5년기대매출_억']=b.매출_억*b.기대연수
b['배수']=b['5년기대매출_억']/b.창업비_억; b['배수_비관']=b.매출_억*b.기대연수_비관/b.창업비_억; b['배수_낙관']=b.매출_억*b.기대연수_낙관/b.창업비_억
b['단순배수(매출/창업비)']=b.매출_억/b.창업비_억
b['순위_단순']=b['단순배수(매출/창업비)'].rank(ascending=False).astype(int); b['순위_위험반영']=b.배수.rank(ascending=False).astype(int)
t=b[['브랜드','서울개업15_22','폐업','매출_억','창업비_억','기대연수','5년기대매출_억','배수','배수_비관','배수_낙관','단순배수(매출/창업비)','순위_단순','순위_위험반영']].sort_values('배수',ascending=False)
pd.set_option('display.width',250); print(t.round(2).to_string(index=False)); t.round(3).to_csv(f'{P}/brand_value.csv',index=False,encoding='utf-8-sig')
print('\n순위가 3계단 이상 바뀐 브랜드:'); print(t[(t.순위_단순-t.순위_위험반영).abs()>=3][['브랜드','순위_단순','순위_위험반영','폐업']].to_string(index=False))
