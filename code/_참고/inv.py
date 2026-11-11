import pandas as pd, glob, os
R='/Users/gw-Kim/post/프로젝트/DAT/프로젝트_막시무스/'
fs=[R+'maximus-analysis-archive/data/raw/'+f for f in ['seoul_commercial_sales_dong_2023.csv','seoul_commercial_stores_dong_2023.csv','seoul_commercial_floating_population_dong.csv','seoul_resident_population_dong.csv','seoul_workplace_population_dong.csv','seoul_attracting_facilities_dong.csv']]+[R+'데이터/원천/seoul_openapi/commercial_area_floating_population.csv',R+'데이터/원천/seoul_openapi/commercial_area_resident_population.csv']
for f in fs:
    for enc in ['utf-8-sig','cp949']:
        try: d=pd.read_csv(f,encoding=enc,low_memory=False); break
        except UnicodeDecodeError: continue
    q=[c for c in d.columns if c in('STDR_YYQU_CD','기준_년분기_코드')]
    print('\n',os.path.basename(f),d.shape); print(list(d.columns)[:12])
    if q: u=sorted(d[q[0]].astype(str).unique()); print('Q',u[0],u[-1],len(u))
    dc=[c for c in d.columns if '행정동_코드' in c and '명' not in c or c=='ADSTRD_CD']
    if dc: print('dongs',d[dc[0]].nunique())
