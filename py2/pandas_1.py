import pandas as pd
import numpy as np
s= pd.Series([1,3,6,np.nan,44,1])
print(s)

dates = pd.date_range('20180101',periods=6)
print(dates)

df = pd.DataFrame(np.random.randn(6,4),index=dates, columns=['a','b','c','d'])
print(df)
df2 = pd.DataFrame(np.arange(12).reshape((3,4)),index=['a','b','c'],columns=['a','b','c','d'])
print(df2)
print(df2.dtypes)
print(df2.describe())
print(df2.T)
print(df2.sort_index(axis=1,ascending=False))
print(df2.sort_values(by='c'))
dates = pd.date_range('20180101',periods=6)
df = pd.DataFrame(np.arange(24).reshape((6,4)),index=dates, columns=['A','B','C','D'])

print(df['A'],df.A)
print(df[0:3],df['20180102':'20180105'])

# select by label:loc
## print(df.loc['20180102'])
print(df.loc['20180102':,['A','B']])
print(df.iloc[[1,3,5],1:3])
# mixed selection:ix

print(df.ix[:4,['A','D']])
print(df)
#print(df[df.A>8])
#df.iloc[2,2] = 1111
#df.loc['20180101','C'] = 0
df.loc[df.A >4,'B'] = 0
print(df)
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
print(df.dropna(axis=1,how='any'))
print(df.isnull().any()==True)



# read_csv from file
data = pd.read_csv('student.csv')
print(data)
data.to_pickle('data.pickle')
