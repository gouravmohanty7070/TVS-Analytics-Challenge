#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df_test = pd.read_csv("test_data.csv")
df_train = pd.read_csv("train_data.csv")
df_test_prediction = pd.read_csv("test_data_prediction.csv")


# In[ ]:


df_test_prediction = df_test_prediction['v39']


# In[ ]:


def drop_columns(df):
    df = df.drop({"v1","v2","v4","v19"}, axis=1)
    
    return df


# In[ ]:


# Fixing categorical data 
def categorical_to_numerical(df):
    cleanup_nums =     {"v5": {"ASC" : 1, "DEALER" : 0},
                        "v11": {"P": 0, "A": 1, "G": 2},
                        "v12": {"P": 0, "A": 1, "G": 2},
                        "v13": {"P": 0, "A": 1, "G": 2},
                        "v14": {"P": 0, "A": 1, "G": 2},
                        "v37": {"N": 0, "Y": 1},
                        "v38": {"NO": 0, "YES": 1}
                        }
    df = df.replace(cleanup_nums)
    
    return df


# In[ ]:


def impute_columns(df_train_imputedValues):
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues["v10"].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v11'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v12'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v13'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v14'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v17'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v18'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v24'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v32'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v33'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v34'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v35'].mode()[0])
    df_train_imputedValues = df_train_imputedValues.fillna(df_train_imputedValues['v36'].mode()[0])
    
    return df_train_imputedValues


# In[ ]:


from scipy.stats import norm, skew 

def fix_skew(df):
    from scipy.special import boxcox1p
    skewed_features = ["v32","v34","v27","v24","v38","v33","v29","v10","v15","v7",]
    lam = 0.15
    for feat in skewed_features:
        #all_data[feat] += 1
        df[feat] = boxcox1p(df[feat], lam)
        
    return df


# In[ ]:


df_train = drop_columns(df_train)
df_train= categorical_to_numerical(df_train)
df_train = impute_columns(df_train)
df_train = fix_skew(df_train)


# In[ ]:


df_test = drop_columns(df_test)
df_test = categorical_to_numerical(df_test)
df_test = impute_columns(df_test)
df_test = fix_skew(df_test)


# In[ ]:


y_train = df_train['v39']
X_train = df_train.drop(columns='v39')
X_test = df_test
y_test = df_test_prediction

print('X train shape: ', X_train.shape)
print('Y train shape: ', y_train.shape)
print('X test shape: ', X_test.shape)
print('Y test shape: ', y_test.shape)


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
# from xgboost import XGBClassifier

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['reg:linear'],
              'learning_rate': [.07], #so called `eta` value
              'max_depth': [9],
              'min_child_weight': [1],
              'silent': [1],
              'subsample': [1],
              'colsample_bytree': [0.5],
              'n_estimators': [1500]}

xgb_grid = GridSearchCV(xgb,
                        parameters,
                        cv = 2,
                        n_jobs = 5,
                        verbose=True)

xgb_grid.fit(X_train,
         y_train)

print(xgb_grid.best_score_)
print(xgb_grid.best_params_) 


# In[ ]:


y_pred=xgb_grid.predict(X_test)

from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error

print("\t\tError Table")
print('Mean Absolute Error      : ', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared  Error      : ', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared  Error : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('R Squared Error          : ', metrics.r2_score(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




