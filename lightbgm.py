
import time
import numpy as np
import pandas as pd
import lightgbm as LGB
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, \
    mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

train_df=pd.read_csv('C:/Users/22454/Desktop/code/RUL-and-SOH-estimation-of-Lithium-ion-satellite-power-systems-using-support-vector-regression-master/B0005.csv',encoding='gbk', index_col=0)
test_df=pd.read_csv('C:/Users/22454/Desktop/code/RUL-and-SOH-estimation-of-Lithium-ion-satellite-power-systems-using-support-vector-regression-master/B0006.csv',encoding='gbk',index_col=0)
print('训练集的数据⼤⼩：',train_df.shape)
print('测试集的数据⼤⼩：',test_df.shape)
print('-'*30)
print('训练集的数据类型：')
print(train_df.dtypes)
print('-'*30)
print(test_df.dtypes)
train_df.head()

print(train_df.isnull().sum())
print('-'*30)
print(test_df.isnull().sum())
#可以看到 数据很密集
#----------------查数据相关性----------------
print('-'*30)
print('查看训练集中数据的相关性')
print(train_df.corr())
print(test_df.corr())

X = train_df['cycle'].values
y = train_df['capacity'].values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

#定义回归模型评估误差指标
def median_absolute_percentage_error(y_true,y_pred):
    return np.median(np.abs((y_pred-y_true)/y_true))
def regression_metrics(true,pred):
    print('回归模型评估指标结果:')
    print('均方误差【MSE】:', mean_squared_error(true, pred))
    print('均方根误差【RMSE】:',np.sqrt(mean_squared_error(true,pred)))
    print('平均绝对误差【MAE】:',mean_absolute_error(true,pred))
    print('绝对误差中位数【MedianAE】:',median_absolute_error(true,pred))
    print('平均绝对百分比误差【MAPE】:',mean_absolute_percentage_error(true,pred))
    print('绝对百分比误差中位数【MedianAPE】:',median_absolute_percentage_error(true,pred))

#建立LGB的dataset格式数据
X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)


#定义超参数dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_depth': 7,
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
'feature_pre_filter':False
}
lgb_train = LGB.Dataset(X_train, y_train)
lgb_eval = LGB.Dataset(X_test, y_test, reference=lgb_train)

#定义callback回调
callback=[LGB.early_stopping(stopping_rounds=10,verbose=True),
          LGB.log_evaluation(period=10,show_stdv=True)]
valid_sets=[lgb_train,lgb_eval]
# 训练 train
m1 = LGB.train(params,lgb_train,num_boost_round=2000,
               valid_sets=[lgb_train,lgb_eval],
               callbacks=callback)
#预测数据集
y_pred = m1.predict(X_test)
#评估模型
regression_metrics(y_test,y_pred)




#模型优化
objective=['regression_l2','regression_l1','quantile','poisson','mape']
metrics=['l2','mae','quantile','poisson','mape']
metrics_test_data=pd.DataFrame(columns=['objective','metric','MAPE','Median APE','MAE'])
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
              '开始目标函数与评估函数评估')
for i in objective:
    for k in metrics:
        size=metrics_test_data.size
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': i,
            'metric':k,
            'max_depth': 7,
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        callback=LGB.early_stopping(stopping_rounds=10,verbose=0)
        gbm = LGB.train(params,lgb_train,num_boost_round=2000,
                valid_sets=lgb_eval,callbacks=[callback])
        y_pred = gbm.predict(X_test)
        metrics_test_data.loc[size]=[i,k,mean_absolute_percentage_error(y_test,y_pred),
                                     median_absolute_percentage_error(y_test,y_pred),
                                     mean_absolute_error(y_test,y_pred)
                                    ]
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),i,'+',k,' 完成评估',
              ' best iteration is:',gbm.best_iteration)


from sklearn.metrics import make_scorer
neg_median_absolute_percentage_error=make_scorer(median_absolute_percentage_error, greater_is_better=False)

#开始gridsearch
from sklearn.model_selection import train_test_split, GridSearchCV
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'开始树深度和叶子结点数GridSearch')
model_lgb = LGB.LGBMRegressor(objective='regression_l1',
                              metric='quantile',
                              learning_rate=0.1,
                              subsample = 0.8,
                              colsample_bytree = 0.8,
                              subsample_freq = 5)
params_test1={
    'max_depth': range(7,11,1),
    'num_leaves':range(30,50,1)
}
gsearch1 = GridSearchCV(estimator=model_lgb,
                        param_grid=params_test1,
                        scoring=neg_median_absolute_percentage_error,
                        cv=5,
                        verbose=1,
                        n_jobs=-1)
X = X.reshape(-1, 1)
# y = y.reshape(-1, 1)
gsearch1.fit(X, y)
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),'完成树深度和叶子结点数GridSearch')
print('Best parameters found by grid search are:', gsearch1.best_params_)

params2 = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'max_depth': 7,
    'num_leaves': 10,
    'learning_rate': 0.1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,

    'min_child_samples': 19,
    'min_child_weight': 0.001,
    'colsample_bytree': 0.7,
    'subsample': 0.9,
    'reg_alpha': 0.3,
    'reg_lambda': 0.03,
     'feature_pre_filter':False

}
lgb_train2 = LGB.Dataset(X_train, y_train)
lgb_eval2 = LGB.Dataset(X_test, y_test, reference=lgb_train)

# 定义callback回调
callback = [LGB.early_stopping(stopping_rounds=10, verbose=True),
            LGB.log_evaluation(period=10, show_stdv=True)]
valid_sets = [lgb_train2, lgb_eval2]
# 训练 train
m2 = LGB.train(params2, lgb_train, num_boost_round=2000,
               valid_sets=[lgb_train2, lgb_eval2],
               callbacks=callback)
# 预测数据集
y_pred = m2.predict(X_test)
# 评估模型
regression_metrics(y_test, y_pred)
fig, ax = plt.subplots(1, figsize=(12, 8))

ax.plot(X, y, color='red', label='old')
ax.plot(X, m2.predict(X), color='blue', label='new')
# ax.set(xlabel='Discharge cycles', ylabel='Capacity/Ah', title='Capacity degradation at ambient temperature of 43°C')
# plt.legend()
# plt.show()
# #数据太少，丑的一逼