# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 17:38:25 2018
# 数据观测与数据预处理
@author: 40393
"""
import pandas as pd
import numpy as np
import datetime as datetime
import datetime
import sys
import xgboost as xgb
import re
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from pandas import Series,DataFrame
from datetime import timedelta
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error

train_order = pd.read_csv('../ms_AI/AI_risk_train/train_order_info.csv',parse_dates=['time_order'],nrows=200)
train_order['amt_order'] = train_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
train_order['time_order_error'] = train_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                               else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))
train_order['time_order_new'] = train_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                               else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else  pd.lib.NaT ))
see=train_order[['time_order','time_order_error','time_order_new']]

train_target = pd.read_csv('../ms_AI/AI_risk_train/train_target.csv',parse_dates = ['appl_sbm_tm'],nrows=10000)

data.loc[data['DAYS_CREDIT_ENDDATE'] < -40000, 'DAYS_CREDIT_ENDDATE'] = np.nan # 对异常值进行替换
################ 数据观察
train_user = pd.read_csv('../ms_AI/AI_risk_train/train_user_info.csv',parse_dates = ['birthday'],nrows=10000)

see=train_user['id'].drop_duplicates()  # 观察有无重复列
train_user.dtypes                 # 列类型
train_user['sex'].value_counts()  # 列取值及频数
train_user.dtypes.value_counts()  # 列类型 频数
#train_target['target'].astype(int).plot.hist()  # 数值型分类变量直方图 
train_user.select_dtypes('object').apply(pd.Series.nunique, axis = 0)   # object 型列 取值

train_user_code = pd.get_dummies(train_user) # 调用独热编码器 生成哑变量
train_user_code.rename(columns={'sex_男':'sex_man'},inplace = True);   # 对列重命名
bb_agg.columns = ["_".join(f_).upper() for f_ in bb_agg.columns]   # 使用指定字符连接列名
#reset_index(name='sex_') #修改列名称

# all_train[['sex','qq_bound']]=train_user[['sex','qq_bound']].apply(LabelEncoder().fit_transform) # 标签编码器，预先处理 nan
def missing_values_table(df):    #计算出缺失值个数及比例
        # Total missing values
        mis_val = df.isnull().sum()
        
        # Percentage of missing values
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        
        # Make a table with the results
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        
        # Rename the columns
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        
        # Sort the table by percentage of missing descending
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        
        # Print some summary information
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        
        # Return the dataframe with missing information
        return mis_val_table_ren_columns    
missing_values = missing_values_table(train_user)

################### 核密度估计图--KDE
train_credit = pd.read_csv('../ms_AI/AI_risk_train/train_credit_info.csv') #nrows=10000
train_target = pd.read_csv('../ms_AI/AI_risk_train/train_target.csv',parse_dates = ['appl_sbm_tm']) # nrows=10000 parse_dates 转为时间变量
see=train_credit['id'].drop_duplicates()     # id 无重复
train_data = pd.merge(train_target,train_credit,on=['id'],how='left')  #连接两表
train_data['target'].value_counts()            # target 分布情况
correlations = train_data.corr()['target'].sort_values()   # 个特征与target 相关系数
train_data['credit_score']=train_data['credit_score'].map(lambda x:-10 if str(x)=='nan' else x) # 对 credit_score 中'nan' 填充为 -10
train_data['credit_score'].astype(int).plot.hist()        #频率分布直方图
plt.title('credit_score of Client') 
plt.xlabel('credit_score') 
plt.ylabel('Count')


# Set the style of plots     频率分布直方图，bins--切断间隔
plt.style.use('fivethirtyeight')
# Plot the distribution of credit_score 
plt.hist(train_data['credit_score'] , edgecolor = 'k', bins = 25)
plt.title('Credit_score of Client')
plt.xlabel('Credit_score'); 
plt.ylabel('Count')

plt.figure(figsize = (5, 4))

# KDE plot of loans that were repaid on time
sns.kdeplot(train_data.loc[train_data['target'] == 0, 'credit_score'] , label = 'target == 0')

# KDE plot of loans which were not repaid on time
sns.kdeplot(train_data.loc[train_data['target'] == 1, 'credit_score'] , label = 'target == 1')

# Labeling of plot
plt.xlabel('credit_score'); plt.ylabel('Density'); plt.title('Distribution of credit_score');


##################################### 各区间违约率占比
df_credit_score = train_data[['target', 'credit_score']]
# Bin the age data
#ins = np.linspace(200, 600, num = 11)
bins=[-11,350,400,450,500,550,800]
df_credit_score['credit_score_binned'] = pd.cut(df_credit_score['credit_score'] ,bins )

#bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
#df_credit_score['credit_score_binned'] = pd.qcut(df_credit_score['credit_score'] ,bins)   # 按分位数切分

# Group by the bin and calculate averages
credit_score_groups  = df_credit_score.groupby('credit_score_binned').mean()
#credit_score_groups  = df_credit_score.groupby('credit_score_binned').agg(['sum','count','mean'])

plt.figure(figsize = (5, 4))

# Graph the age bins and the average of the target as a bar plot
plt.bar(credit_score_groups.index.astype(str), 100 * credit_score_groups['target'])

# Plot labeling
plt.xticks(rotation = 75); plt.xlabel('credit_score'); plt.ylabel('Failure to Repay (%)')
plt.title('Failure to Repay by credit_score Group');
############# 生成透视表
data = pd.read_csv('../input/bureau_balance.csv')
cut_points = [0,2,4,12,24,36]                         # 设置切断区间
cut_points = cut_points + [data["MONTHS_BALANCE"].max()] # 增加切断上限值
labels = ["2MON","4MON","12MON","24MON","36MON","ABOVE"] # 各区间命名
data["MON_INTERVAL"] = pd.cut(data["MONTHS_BALANCE"], cut_points,labels=labels,include_lowest=True) # 切断
   # 生成透视表，指定转换函数
feature = pd.pivot_table(data,index=["SK_ID_BUREAU"],columns=["MON_INTERVAL"],values=["STATUS"],aggfunc=[np.max,np.mean,np.std]).astype('float32')
feature.columns = ["_".join(f_).upper() for f_ in feature.columns]  # 修改列命


############# 调用算法模型 LGBM

X_train =train_data[['credit_score','overdraft','quota']]
y_train =train_data['target']
X_test  =train_data[['credit_score','overdraft','quota']]
lgb_train = lgb.Dataset(X_train, y_train,) # categorical_feature={'sex'} 分类特征
lgb_test = lgb.Dataset(X_test)
params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':'auc',
        'num_leaves': 25,
        'learning_rate': 0.01,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'min_data_in_leaf':5,
        'max_bin':200,
        'verbose': 0,
}
gbm = lgb.train(params,lgb_train,num_boost_round=200)
predict = gbm.predict(X_test)
minmin = min(predict)
maxmax = max(predict)
predict=Series(predict)
vfunc_lg = predict.map(lambda x:(x-minmin)/(maxmax-minmin))  # 将 LGBM 输出概率值映射至[0,1]区间

gbm.feature_importance(importance_type='split')    # 输出特征重要性
gbm.feature_name()                                 # 特征名称


#################   XGBoost
params = {'booster': 'gbtree',
              'objective':'rank:pairwise',
              'eval_metric' : 'auc',
              'eta': 0.02,
              'max_depth': 5,  # 4 3
              'colsample_bytree': 0.7,#0.8
              'subsample': 0.7,
              'min_child_weight': 1,  # 2 3
              'seed': 1111,
              'silent':1
              }
dtrain = xgb.DMatrix(X_train, label=y_train)
dvali = xgb.DMatrix(X_test)
model = xgb.train(params, dtrain, num_boost_round=800)
predict = model.predict(dvali)
minmin = min(predict)
maxmax = max(predict)
predict=Series(predict)
vfunc_xg = predict.map(lambda x:(x-minmin)/(maxmax-minmin)) 


############# 模型融合 stacking
model_xgb = Regressor(dataset=xgb_dataset, estimator=xgb_feature,name='xgb',use_cache=False)
model_xgb2 = Regressor(dataset=xgb_dataset, estimator=xgb_feature2,name='xgb2',use_cache=False)
model_xgb3 = Regressor(dataset=xgb_dataset, estimator=xgb_feature3,name='xgb3',use_cache=False)
model_lgb = Regressor(dataset=lgb_dataset, estimator=lgb_feature,name='lgb',use_cache=False)
model_gbdt = Regressor(dataset=xgb_dataset, estimator=gbdt_model,name='gbdt',use_cache=False)

ipeline = ModelsPipeline(model_xgb,model_xgb2,model_xgb3,model_lgb,model_gbdt)
stack_ds = pipeline.stack(k=5, seed=111, add_diff=False, full_test=True)
stacker = Regressor(dataset=stack_ds, estimator=LinearRegression,parameters={'fit_intercept': False})
predict_result = stacker.predict()







