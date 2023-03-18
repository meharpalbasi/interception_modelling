#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nfl_data_py as nfl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier


# In[2]:


pbp = nfl.import_pbp_data([2021,2022])


# In[3]:


pbp.columns


# In[4]:


pbp.head(10)


# In[5]:


pd.set_option('display.max_columns', None)
print(pbp)


# In[6]:


print(pbp.shape)


# In[7]:


pbp_clean = pbp[pbp['pass'] == 1 & (pbp['play_type'] != 'no_play')]


# In[8]:


pbp_clean.shape


# In[9]:


sns.countplot(x = pbp_clean['interception'])
plt.show()


# In[10]:


interception = pbp_clean[(pbp_clean['interception']==1)]
sns.countplot(x = interception['down'])
plt.show()


# In[11]:


sacks = pbp_clean[(pbp_clean['interception'] == 1)]
sns.countplot(x=sacks["number_of_pass_rushers"])
plt.show()


# In[12]:


sacks = pbp_clean[(pbp_clean['interception'] == 1)]
sns.countplot(x=sacks["defenders_in_box"])
plt.show()


# In[13]:


pbp_clean['obvious_pass'] = np.where((pbp_clean['down'] == 3) & (pbp_clean['ydstogo'] >= 6), 1,0)


# In[15]:


pre_df = pbp_clean[['game_id', 'play_id', 'season', 'name', 'down', 'ydstogo', 'yardline_100', 'game_seconds_remaining',
                    'defenders_in_box', 'number_of_pass_rushers', 'xpass', 'obvious_pass', 'interception', 'qb_hit', 'no_huddle', 'air_yards']]


# In[16]:


pre_df.isna().sum()


# In[17]:


df = pre_df.dropna()


# In[18]:


df.isna().sum()


# In[19]:


df.head()


# In[20]:


df['down'] = df['down'].astype('category')
df_no_ids = df.drop(columns = ['game_id', 'play_id', 'name', 'season'])
df_no_ids = pd.get_dummies(df_no_ids, columns = ['down'])


# In[21]:


df_no_ids.columns


# In[22]:


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=86)
for train_index, test_index in sss.split(df_no_ids, df_no_ids['interception']):
    strat_train_set = df_no_ids.iloc[train_index]
    strat_test_set = df_no_ids.iloc[test_index]

X_train = strat_train_set.drop(columns = ['interception'])
Y_train = strat_train_set['interception']
X_test = strat_test_set.drop(columns = ['interception'])
Y_test = strat_test_set['interception']


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


LR = LogisticRegression()
LR.fit(X_train, Y_train)

LR_pred = pd.DataFrame(LR.predict_proba(X_test), columns = ['no_interception', 'interception'])[['interception']]

print('Brier Score: ', brier_score_loss(Y_test, LR_pred))


# In[26]:


RF = RandomForestClassifier()
RF.fit(X_train, Y_train)

RF_pred = pd.DataFrame(RF.predict_proba(X_test), columns = ['no_interception', 'interception'])[['interception']]

print('Brier Score: ', brier_score_loss(Y_test, RF_pred))


# In[27]:


XGB = XGBClassifier(objective="binary:logistic", random_state=42)
XGB.fit(X_train, Y_train)

XGB_pred = pd.DataFrame(XGB.predict_proba(X_test), columns = ['no_interception', 'interception'])[['interception']]

print('Brier Score: ', brier_score_loss(Y_test, XGB_pred))


# In[28]:


sorted_idx = XGB.feature_importances_.argsort()
plt.barh(X_train.columns[sorted_idx], XGB.feature_importances_[sorted_idx])
plt.title("XGBClassifier Feature Importance")
plt.show()


# In[30]:


make_interception_preds = df_no_ids.drop('interception', axis = 1)
XGB_total_predictions = pd.DataFrame(XGB.predict_proba(make_interception_preds), columns = ['no_interception', 'interception_pred'])[['interception_pred']]

interception_preds = df.reset_index().drop(columns = ['index'])
interception_preds['interception_pred'] = XGB_total_predictions

interception_preds['interception_oe'] = interception_preds['interception'] - interception_preds['interception_pred']
interception_preds[(interception_preds['season'] == 2022)].groupby('name').agg({'interception': 'sum', 'interception_pred': 'sum', 'interception_oe': 'sum'}).reset_index().sort_values('interception_oe', ascending = True)


# In[32]:


sns.boxplot( x=interception_preds["interception"], y=interception_preds["interception_pred"])
plt.show()


# In[ ]:




