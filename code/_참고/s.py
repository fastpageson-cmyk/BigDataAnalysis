import pandas as pd,sys
x=pd.read_csv(sys.argv[1],encoding='cp949',usecols=[0,1,2,3,4,5],encoding_errors='replace',low_memory=False); print(x.columns.tolist()); q=x.columns[0]
print(x.groupby(q).size().to_string()); print(x[x.iloc[:,3].astype(str).str.startswith('CS1')].groupby(q)[x.columns[5]].sum().div(1e12).round(2).to_string())
