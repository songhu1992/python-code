# -*- coding: utf-8 -*-
#源：https://github.com/chenkkkk/User-loan-risk-prediction/blob/master/model1/04094.py
"""
Created on Wed Dec  5 09:49:59 2018
@author: songhu
"""
import pandas as pd
import numpy as np
import datetime as datetime
import datetime
import sys
import xgboost as xgb
import re

from pandas import Series,DataFrame
from datetime import timedelta
from dateutil.parser import parse
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import auc

################ 数据观察
train_bankcard = pd.read_csv('../ms_AI/AI_risk_train/train_bankcard_info.csv',nrows=10000)
train_bankcard['id'].unique()
train_bankcard['id'].isnull()
################ 数据预处理
train_auth = pd.read_csv('../ms_AI/AI_risk_train/train_auth_info.csv',parse_dates=['auth_time'],nrows=10000)
auth_idcard_df = pd.DataFrame();
auth_idcard_df['id'] = train_auth['id'];
auth_idcard = train_auth['id_card'].map(lambda x:0 if str(x)=='nan' else 1)  #  0:空值，1：非空
#auth_idcard = train_auth['id_card'].isnull().map(lambda x:0 if str(x)=='True' else 1)
auth_time = train_auth['auth_time'].map(lambda x:0 if str(x)=='NaT' else 1)
auth_phone = train_auth['phone'].map(lambda x:0 if str(x)=='nan' else 1)
auth_idcard_df['auth_idcard_df'] = auth_idcard
auth_idcard_df['auth_time_df'] = auth_time
auth_idcard_df['auth_phone_df'] = auth_phone

train_bankcard = pd.read_csv('../ms_AI/AI_risk_train/train_bankcard_info.csv',nrows=10000)
train_bankcard_bank_count = train_bankcard.groupby(by=['id'], as_index=False)['bank_name'].agg({'bankcard_count':lambda x :len(set(x))})
train_bankcard_card_count = train_bankcard.groupby(by=['id'], as_index=False)['card_type'].agg({'card_type_count':lambda x :len(set(x))})
train_bankcard_phone_count = train_bankcard.groupby(by=['id'], as_index=False)['phone'].agg({'phone_count':lambda x :len(set(x))})

# 生成评分反序  共线性
train_credit = pd.read_csv('../ms_AI/AI_risk_train/train_credit_info.csv',nrows=10000)
train_credit['credit_score_inverse'] = train_credit['credit_score'].map(lambda x :605-x)
train_credit['can_use'] = train_credit['quota'] - train_credit['overdraft']   # 额度-使用值

train_order = pd.read_csv('../ms_AI/AI_risk_train/train_order_info.csv',parse_dates=['time_order'],nrows=10000)
train_order['amt_order'] = train_order['amt_order'].map(lambda x:np.nan if ((x == 'NA')| (x == 'null')) else float(x))
train_order['time_order'] = train_order['time_order'].map(lambda x : pd.lib.NaT if (str(x) == '0' or x == 'NA' or x == 'nan')
                                else (datetime.datetime.strptime(str(x),'%Y-%m-%d %H:%M:%S') if ':' in str(x)
                                else (datetime.datetime.utcfromtimestamp(int(x[0:10])) + datetime.timedelta(hours = 8))))
train_order_time_max = train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_max':lambda x:max(x)})
train_order_time_min = train_order.groupby(by=['id'], as_index=False)['time_order'].agg({'train_order_time_min':lambda x:min(x)})
train_order_type_zaixian = train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='在线支付').values].count()).reset_index(name = 'type_pay_zaixian')
#train_order['type_pay_zaixian']=train_order['type_pay'].map(lambda x: 1 if re.match('在线',str(x)) else 0)
train_order_type_huodao = train_order.groupby(by=['id']).apply(lambda x:x['type_pay'][(x['type_pay']=='货到付款').values].count()).reset_index(name = 'type_pay_huodao')

train_recieve = pd.read_csv('../ms_AI/AI_risk_train/train_recieve_addr_info.csv',nrows=10000)
train_recieve['region'] = train_recieve['region'].map(lambda x:str(x)[:2])    #取出地址的前两位
tmp_tmp_recieve = pd.crosstab(train_recieve.id,train_recieve.region)   # 生成交叉表
tmp_tmp_recieve = tmp_tmp_recieve.reset_index()                   # 重新定义新的索引
tmp_tmp_recieve_phone_count = train_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].count())
    ## = train_recieve['fix_phone'].groupby(train_recieve['id']).count().reset_index()
tmp_tmp_recieve_phone_count=tmp_tmp_recieve_phone_count.reset_index()
tmp_tmp_recieve_phone_count_unique = train_recieve.groupby(by=['id']).apply(lambda x:x['fix_phone'].nunique())   # nunique 查看不同手机号码个数
tmp_tmp_recieve_phone_count_unique=tmp_tmp_recieve_phone_count_unique.reset_index()

train_target = pd.read_csv('../ms_AI/AI_risk_train/train_target.csv',parse_dates = ['appl_sbm_tm'],nrows=10000)

train_user = pd.read_csv('../ms_AI/AI_risk_train/train_user_info.csv',parse_dates = ['birthday'],nrows=10000)
is_hobby = train_user['hobby'].map(lambda x:0 if str(x)=='nan' else 1)
is_hobby_df = pd.DataFrame()
is_hobby_df['id'] = train_user['id']
is_hobby_df['is_hobby'] = is_hobby
is_idcard = train_user['id_card'].map(lambda x:0 if str(x)=='nan' else 1)
is_idcard_df = pd.DataFrame()
is_idcard_df['id'] = train_user['id']
is_idcard_df['is_idcard'] = is_idcard


 #usesr_birthday
tmp_tmp = train_user[['id','birthday']]
tmp_tmp = tmp_tmp.set_index(['id'])
is_double_ = tmp_tmp['birthday'].map(lambda x:(str(x) == '--')*1).reset_index(name='is_double_')
is_0_0_0 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0-0-0')*1).reset_index(name='is_0_0_0')
is_1_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '1-1-1')*1).reset_index(name='is_1_1_1')
is_0000_00_00 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0000-00-00')*1).reset_index(name='is_0000_00_00')
is_0001_1_1 = tmp_tmp['birthday'].map(lambda x:(str(x) == '0001-1-1')*1).reset_index(name='is_0001_1_1')
is_hou_in = tmp_tmp['birthday'].map(lambda x:('后' in str(x))*1).reset_index(name='is_hou_in')
# 对以 -0 和-00 结尾日期 取 NaT
train_user['birthday'] = train_user['birthday'].map(lambda x:pd.lib.NaT if x.endswith('-0') or x.endswith('-00') else x)
train_user['birthday'] = train_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) else pd.lib.NaT)
#train_user['birthday_error'] = train_user['birthday'].map(lambda x:datetime.datetime.strptime(str(x),'%Y-%m-%d') if(re.match('19\d{2}-\d{1,2}-\d{1,2}',str(x)) and '-0' not in str(x)) else pd.lib.NaT)
#see1=train_user[['birthday_new','birthday_error','birthday']] 

train_data = pd.merge(train_target,train_auth,on=['id'],how='left')
train_data = pd.merge(train_data,train_user,on=['id'],how='left')
train_data = pd.merge(train_data,train_credit,on=['id'],how='left')

train_data['hour'] = train_data['appl_sbm_tm'].map(lambda x:x.hour)
train_data['month'] = train_data['appl_sbm_tm'].map(lambda x:x.month)
train_data['year'] = train_data['appl_sbm_tm'].map(lambda x:x.year)
train_data['quota_use_ratio'] = train_data['overdraft'] / (train_data['quota']+0.01)     # 使用值/额度值
train_data['nan_num'] = train_data.isnull().sum(axis=1)  # 计算每行 nan 个数
train_data['diff_day'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['auth_time']).days,axis=1)  # 用户贷款申请的提交时间-认证时间
see2=train_data[['appl_sbm_tm','auth_time','diff_day']]
train_data['how_old'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['birthday']).days/365,axis=1) #  申请日期-出生日期
see3=train_data[['appl_sbm_tm','birthday','how_old']]

train_bankcard_phone_list = train_bankcard.groupby(by=['id'])['phone'].apply(lambda x:(set(x.tolist()))).reset_index(name = 'bank_phone_list')
train_data = pd.merge(train_data,train_bankcard_phone_list,on=['id'],how='left')

#train_data['exist_phone'] = train_data.apply(lambda x:x['phone'] in x['bank_phone_list'],axis=1)
#train_data['exist_phone'] = train_data['exist_phone']*1
train_data = train_data.drop(['bank_phone_list'],axis=1)
   #bankcard_info  
   #
bank_name = train_bankcard.groupby(by= ['id'], as_index= False)['bank_name'].agg({'bank_name_len':lambda x:len(set(x))})
bank_num = train_bankcard.groupby(by= ['id'], as_index = False)['tail_num'].agg({'tail_num_len':lambda x:len(set(x))})
bank_phone_num = train_bankcard.groupby(by= ['id'], as_index = False)['phone'].agg({'bank_phone_num':lambda x:x.nunique()})


train_data = pd.merge(train_data,bank_name,on=['id'],how='left')
train_data = pd.merge(train_data,bank_num,on=['id'],how='left')

train_data = pd.merge(train_data,train_order_time_max,on=['id'],how='left')
train_data = pd.merge(train_data,train_order_time_min,on=['id'],how='left')
train_data = pd.merge(train_data,train_order_type_zaixian,on=['id'],how='left')
train_data = pd.merge(train_data,train_order_type_huodao,on=['id'],how='left')
train_data = pd.merge(train_data,is_double_,on=['id'],how='left')
train_data = pd.merge(train_data,is_0_0_0,on=['id'],how='left')
train_data = pd.merge(train_data,is_1_1_1,on=['id'],how='left')
train_data = pd.merge(train_data,is_0000_00_00,on=['id'],how='left')
train_data = pd.merge(train_data,is_0001_1_1,on=['id'],how='left')
train_data = pd.merge(train_data,is_hou_in,on=['id'],how='left')
# train_data = pd.merge(train_data,is_nan,on=['id'],how='left')
train_data = pd.merge(train_data,tmp_tmp_recieve,on=['id'],how='left')
train_data = pd.merge(train_data,tmp_tmp_recieve_phone_count,on=['id'],how='left')
train_data = pd.merge(train_data,tmp_tmp_recieve_phone_count_unique,on=['id'],how='left')
train_data = pd.merge(train_data,bank_phone_num,on=['id'],how='left')
train_data = pd.merge(train_data,is_hobby_df,on=['id'],how='left')
train_data = pd.merge(train_data,is_idcard_df,on=['id'],how='left')
train_data = pd.merge(train_data,auth_idcard_df,on=['id'],how='left')
#train_data = pd.merge(train_data,auth_phone_df,on=['id'],how='left')

#    "增加特征"
#    train_data = pd.merge(train_data,train_bankcard_bank_count,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_bankcard_card_count,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_bankcard_phone_count,on=['id'],how='left')
   
#    train_data = pd.merge(train_data,auth_time_df,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_mean_unit_price,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_mean_amt_order,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_phone_unique,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_many_success,on=['id'],how='left')
#    train_data = pd.merge(train_data,train_order_many_occuer,on=['id'],how='left')
train_data['day_order_max'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_max']).days,axis=1);train_data = train_data.drop(['train_order_time_max'],axis=1)
train_data['day_order_min'] = train_data.apply(lambda row: (row['appl_sbm_tm'] - row['train_order_time_min']).days,axis=1);train_data = train_data.drop(['train_order_time_min'],axis=1)
    #order_info
order_time = train_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_time':len})
order_mean = train_order.groupby(by = ['id'],as_index=False)['amt_order'].agg({'order_mean':np.mean})
unit_price_mean = train_order.groupby(by = ['id'],as_index=False)['unit_price'].agg({'unit_price_mean':np.mean})
order_time_set = train_order.groupby(by = ['id'],as_index=False)['time_order'].agg({'order_time_set':lambda x:len(set(x))})

#   "4_19"
    # _loan = pd.merge(train_order[['time_order','amt_order','id']], train_target[['appl_sbm_tm','id']],on=['id'],how='right')
    # before_loan = _loan[_loan.time_order<=_loan.appl_sbm_tm]
    # after_loan = _loan[_loan.time_order>_loan.appl_sbm_tm]
    # before_loan_time = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_time':len})
    # after_loan_time = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_time':len})
    # before_loan_mean = before_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'before_loan_mean':np.mean})
    # after_loan_mean = after_loan.groupby(by=['id'],as_index=False)['amt_order'].agg({'after_loan_mean':np.mean})
    # train_data = pd.merge(train_data,before_loan_time,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_loan_time,on=['id'],how='left')
    # train_data = pd.merge(train_data,before_loan_mean,on=['id'],how='left')
    # train_data = pd.merge(train_data,after_loan_mean,on=['id'],how='left')
train_data = pd.merge(train_data,order_time,on=['id'],how='left')
train_data = pd.merge(train_data,order_mean,on=['id'],how='left')
train_data = pd.merge(train_data,order_time_set,on=['id'],how='left')
train_data = pd.merge(train_data,unit_price_mean,on=['id'],how='left')

dummy_fea = ['sex', 'merriage', 'income', 'qq_bound', 'degree', 'wechat_bound','account_grade','industry']
dummy_df = pd.get_dummies(train_data.loc[:,dummy_fea])
train_data_copy = pd.concat([train_data,dummy_df],axis=1)
train_data_copy = train_data_copy.fillna(0)
vaild_train_data = train_data_copy.drop(dummy_fea,axis=1)
valid_train_train = vaild_train_data[vaild_train_data.appl_sbm_tm < datetime.datetime(2017,4,1)]
valid_train_test = vaild_train_data[vaild_train_data.appl_sbm_tm >= datetime.datetime(2017,4,1)]
valid_train_train = valid_train_train.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
valid_train_test = valid_train_test.drop(['appl_sbm_tm','id','id_card_x','auth_time','phone','birthday','hobby','id_card_y'],axis=1)
vaild_train_x = valid_train_train.drop(['target'],axis=1)
vaild_test_x = valid_train_test.drop(['target'],axis=1)
#redict_result, modelee = xgb_feature(vaild_train_x,valid_train_train['target'].values,vaild_test_x,None)
#print('valid auc',roc_auc_score(valid_train_test['target'].values,redict_result))
#sys.exit(23)










