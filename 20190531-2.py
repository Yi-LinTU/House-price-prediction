
# coding: utf-8

# In[1]:

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import warnings
warnings.filterwarnings('ignore')


# In[2]:

data_train=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')
data_train


# In[3]:

data_train.shape,data_test.shape


# In[4]:

data_train.info()


# In[5]:

#删除第一列['building_id']
data_train.drop(['building_id'],axis =1)


# In[6]:

#求相關性矩陣，篩選與total_price具有最大係數的相關變量
corr_matrix=data_train.corr()
f,ax=plt.subplots(figsize=(20,9))
sns.heatmap(corr_matrix,vmax=1,annot=True)



# In[7]:

#選取具有最大相關性的特點，top_corr_feature
top_corr_feature=corr_matrix.index[abs(corr_matrix["total_price"])>0.05]
print(top_corr_feature)


# In[8]:

#進一步縮小相關矩陣
top_corr_matrix = data_train[top_corr_feature].corr()
f,ax = plt.subplots(figsize=(15,10))
print(top_corr_matrix)
sns.heatmap(top_corr_matrix,vmax=1,annot=True)


# In[9]:

#求數據兩兩之間關係，繪製關係點圖
cols=['parking_area', 'parking_price', 'land_area', 'building_area',
       'village_income_median', 'doc_rate', 'master_rate', 'bachelor_rate',
       'jobschool_rate', 'highschool_rate', 'junior_rate', 'elementary_rate',
       'II_10000', 'III_5000', 'III_10000', 'V_500', 'V_1000', 'V_5000',
       'V_10000', 'VI_5000', 'VI_10000', 'VII_500', 'VII_1000', 'VII_5000',
       'VII_10000', 'VIII_1000', 'VIII_5000', 'VIII_10000', 'IX_1000',
       'IX_5000', 'IX_10000', 'X_1000', 'X_5000', 'X_10000', 'XI_5000',
       'XI_10000', 'XIII_5000', 'XIII_10000']
#sns.pairplot(data_train[cols],size=4)
#plt.show()


# In[10]:

data_train[cols].isnull().sum()


# In[11]:

data_test[cols].isnull().sum()


# In[12]:


print(data_train['parking_area'].median())         #parking_area中位數
print(data_train['parking_price'].median())          #parking_price中位數
print(data_train['village_income_median'].mean())          #village_income_median中位數

print(data_test['parking_area'].median())         #parking_area中位數
print(data_test['parking_price'].median())          #parking_price中位數
print(data_test['village_income_median'].mean())          #village_income_median中位數


# In[13]:

# data_train['parking_area'].fillna(5.758023141256257, inplace = True)
# data_train['parking_price'].fillna(43791.94714143084, inplace = True)
data_train['village_income_median'].fillna(642.0, inplace = True)

# data_test['parking_area'].fillna(5.758023141256257, inplace = True)
# data_test['parking_price'].fillna(43791.94714143084, inplace = True)
data_test['village_income_median'].fillna(641.0, inplace = True)


# In[14]:

data_train[cols].isnull().sum()


# In[15]:

data_test[cols].isnull().sum()


# In[16]:

#計算缺失值所佔比例，允許missing_data的數目，先定25％

train_nas=data_train.isnull().sum().sort_values(ascending=False)
percent=(data_train.isnull().sum()/data_train.isnull().count()).sort_values(ascending=False)
missing_data=pd.concat([train_nas,percent],axis=1,keys=['train_nas','percent'])
missing_data.head(20)



# In[17]:


#對預測數據total_price進行視覺化
data_train['total_price'].describe()



# In[18]:

from scipy import stats
from scipy.stats import norm,skew

#sns.distplot(data_train['total_price'],fit=norm)
(mu,sigma)=norm.fit(data_train['total_price'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],
            loc='best')
plt.ylabel('Frequency')
plt.title('total_price distribution')
fig = plt.figure(figsize=(8,5))



# In[19]:


# res = stats.probplot(data_train['total_price'], plot=plt)
# #正態概率圖，用來檢驗數據是否服從正態分佈，如果是一條直線的話，表示服從正態分佈
# plt.show()


# In[20]:

print("Skewness:%f"%data_train['total_price'].skew())
print("Kurtosis:%f"%data_train['total_price'].kurt())


# In[21]:

data_train.total_price=np.log1p(data_train.total_price)#取對數，使其符合正態分佈


# In[22]:

fig=plt.figure(figsize=(15,5))
plt.subplot(121)
y=data_train.total_price
sns.distplot(y,fit=norm)
plt.ylabel('Frequency')
plt.xlabel('total_price')
plt.title('total_price Distribution')

plt.subplot(122)
res=stats.probplot(data_train['total_price'],plot=plt)


# In[23]:

#用線性模型進行預測：LinearRegression,Ridge,Lasso
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score #模型準確度評價指標 R2
from sklearn.model_selection import train_test_split

cols=['building_complete_dt', 'parking_way',
       'land_area', 'building_area', 'town', 
       'village_income_median', 'town_population_density', 'doc_rate',
       'master_rate', 'bachelor_rate', 'jobschool_rate', 'highschool_rate',
       'junior_rate', 'elementary_rate', 'born_rate', 'divorce_rate', 'N_500',
       'N_1000', 'I_1000', 'I_5000', 'I_10000', 'II_500', 'II_1000', 'II_5000',
       'II_10000', 'III_250', 'III_500', 'III_1000', 'III_5000', 'III_10000',
       'IV_index_500', 'IV_1000', 'IV_5000', 'IV_10000', 'IV_MIN', 'V_100',
       'V_250', 'V_500', 'V_1000', 'V_5000', 'V_10000', 'VI_250', 'VI_500',
       'VI_index_500', 'VI_1000', 'VI_5000', 'VI_10000', 'VI_MIN', 'VII_50',
       'VII_100', 'VII_250', 'VII_500', 'VII_1000', 'VII_5000', 'VII_10000',
       'VIII_250', 'VIII_500', 'VIII_1000', 'VIII_5000', 'VIII_10000',
       'IX_100', 'IX_250', 'IX_500', 'IX_1000', 'IX_5000', 'IX_10000', 'X_250',
       'X_500', 'X_1000', 'X_5000', 'X_10000', 'XI_250', 'XI_500', 'XI_1000',
       'XI_5000', 'XI_10000', 'XII_250', 'XII_500', 'XII_1000', 'XII_5000',
       'XII_10000', 'XIII_500', 'XIII_1000', 'XIII_index_1000', 'XIII_5000',
       'XIII_10000', 'XIV_10000']


x = data_train[cols].values
y = data_train['total_price'].values

X_train1,X_test1, y_train1, y_test1 = train_test_split(x,y, test_size=0.3, random_state=42)

Regs={
    'LinearRegression':LinearRegression(),
    'ridge':Ridge(),
    'Lasso':Lasso(alpha =0.001, random_state=1)
}
for Reg in Regs:
    try:
        Regs[Reg].fit(X_train1,y_train1)
        y_pred1=Regs[Reg].predict(X_test1)
        print(Reg+" cost:"+str(r2_score(y_test1,y_pred1)))
    except Exception as e:
        print(Reg+"Error")
        print(str(e)) 

'''
#也可以使用最簡單的分步計算
line.fit(X_train1,y_train1)
ridge.fit(X_train1,y_train1)
lasso.fit(X_train1,y_train1)
#預測數據
line_y_pre=line.predict(x_test1)
ridge_y_pre=ridge.predict(x_test1)
lasso_y_pre=lasso.predict(x_test1)
'''
'''
line_score=r2_score(y_test,line_y_pre)
ridge_score=r2_score(y_test,ridge_y_pre)
lasso_score=r2_score(y_test,lasso_y_pre)
display(line_score,ridge_score,lasso_score)
'''


# In[24]:

# Reg1=Lasso()
# Reg1.fit(X_train1,y_train1)
# y_pred1=Reg1.predict(X_test1)
# print(y_pred1)


# In[25]:

# y_pred_L=np.expm1(y_pred1)
# y_pred_L.shape()


# In[26]:

# sum(abs(y_pred1-y_test1)/len(y_pred1))


# In[27]:

# prediction1 = pd.DataFrame(y_pred_L, columns=['total_price'])
# result1 = pd.concat([data_test['building_id'], prediction1], axis=1)
# result1


# In[28]:

from sklearn import preprocessing
from sklearn import linear_model,svm,gaussian_process
from sklearn.ensemble import RandomForestRegressor

import numpy as np


# In[29]:

x_scaler = preprocessing.StandardScaler().fit_transform(x)
y_scaler = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaler, y_scaler, test_size=0.3, random_state=42)


# In[30]:

clfs={
#    'svm':svm.SVR(),
    'RandomForestRegressor':RandomForestRegressor(n_estimators=1000),
#    'BayesianRidge':linear_model.BayesianRidge()
    'Lasso':Lasso(alpha =0.001, random_state=1)
}
for clf in clfs:
    try:
        clfs[clf].fit(X_train,y_train)
        y_pred=clfs[clf].predict(X_test)
        print(clf+"cost:"+str(np.sum(y_pred-y_test)/len(y_pred)))
    except Exception as e:
        print(clf+"Error")
        print(str(e))


# In[31]:

X_train,X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)


# In[32]:

clf=RandomForestRegressor(n_estimators=1000)
#clf=Lasso(alpha =0.001, random_state=1)
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)


# In[33]:

# 預測結果和正確結果差異
sum(abs(y_pred-y_test)/len(y_pred))


# In[34]:

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

rfr = clf


# In[35]:

x = data_test[cols].values
y_rfr_pred = rfr.predict(x)
print(y_rfr_pred)

print(y_rfr_pred.shape)


# In[36]:

#預測得到的數據需要inverse
y_rfr_pred1=np.expm1(y_rfr_pred)


# In[37]:

prediction = pd.DataFrame(y_rfr_pred1, columns=['total_price'])
result = pd.concat([ data_test['building_id'], prediction], axis=1)
result.columns


# In[38]:

result


# In[39]:

result.to_csv('./Predictions.csv', index=False)


# In[ ]:



