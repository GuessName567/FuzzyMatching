# -*- coding: utf-8 -*-
"""
Created on Sun May  8 07:36:38 2022

@author: luguangli: RF to predict matching quality
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  2 21:34:14 2022

@author: Yunchu Liao, Guangli Lu
"""

import os
import numpy as np
import pandas as pd
import scipy
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


#%%
"""get relevant variables and briefly clean"""

df = pd.read_pickle(yourdataset)

xlist=[a list of variables]
yvar='matched' 
###generate indicators for missing variables and fill original missing values with 0
threshold=int(df.shape[0]/20)
for x in xlist:
    if df[x].isnull().astype(float).sum()>threshod:
        df[x+'_miss']=df[x].isnull().astype(float)
df[xlist]=df[xlist].fillna(0)

#%%
"""get relevant training data: in my dataset, some of the data are labeled, while others are not. The labeled are used to train/evaluate model. This doesn't necessarily apply to you"""
labeled=df[df.matched.notna()]

x= labeled[xlist]
y= pd.DataFrame(labeled[yvar])

seed =5
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3, random_state=seed) #test_size是训练集数据占比，random_state任取


#%%
"""grid search for optimal parameters"""
rfc=RandomForestClassifier() #initiate random forest with the optimal parameter above

rfc.get_params()
param_test1 = {'n_estimators': range(10,400,10)}
gsearch1 = GridSearchCV(estimator = RandomForestClassifier(min_samples_split=100, 
                                                           min_samples_leaf=20,
                                                           max_depth=8, random_state=10), #一定要注意加随机种子
                        param_grid = param_test1, 
                        scoring='roc_auc', #scoring评估方法
                        cv=5) #交叉验证的折数
gsearch1.fit(xtrain, ytrain)
print(gsearch1.best_params_, gsearch1.best_score_) #gsearch1.best_params_最好的参数取值是什么，gsearch1.best_score_最好参数取值下roc_acu是什么
                        

param_test2 = {'min_samples_split':range(20, 200, 20), 'min_samples_leaf':range(10, 110, 10)} #这里是需要调的参数
gsearch2 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=gsearch1.best_estimator_.get_params()['n_estimators'], #此处使用上方已经得到的最优参数
                                                           max_depth=8,n_jobs=20, random_state=10), 
                        param_grid = param_test2, 
                        scoring='roc_auc',
                        cv=5)
gsearch2.fit(xtrain,ytrain)
print(gsearch2.best_params_, gsearch2.best_score_)

param_test3 = {'max_depth':range(3, 30, 2)} 

gsearch3 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=gsearch1.best_estimator_.get_params()['n_estimators'], 
                                                           min_samples_split=gsearch2.best_estimator_.get_params()['min_samples_split'], 
                                                           min_samples_leaf=gsearch2.best_estimator_.get_params()['min_samples_leaf'],n_jobs=20,
                                                           random_state=10), 
                        param_grid = param_test3, 
                        scoring='roc_auc',
                        cv=5)
gsearch3.fit(xtrain,ytrain)
print(gsearch3.best_params_, gsearch3.best_score_)

param_test4 = {'criterion':['gini', 'entropy'], 'class_weight':[None, 'balanced']} #此处调成balanced可以防止样本不平衡的情况
gsearch4 = GridSearchCV(estimator = RandomForestClassifier(n_estimators=gsearch1.best_estimator_.get_params()['n_estimators'],
                                                           max_depth=gsearch3.best_estimator_.get_params()['max_depth'], 
                                                           min_samples_split=gsearch2.best_estimator_.get_params()['min_samples_split'],
                                                           min_samples_leaf=gsearch2.best_estimator_.get_params()['min_samples_leaf'],n_jobs=20,
                                                           random_state=10), 
                        param_grid = param_test4, 
                        scoring='roc_auc',
                        cv=5)
gsearch4.fit(xtrain,ytrain)
print(gsearch4.best_params_, gsearch4.best_score_)


print(gsearch1.best_params_,gsearch2.best_params_,gsearch3.best_params_,gsearch4.best_params_)

#%%
# {'n_estimators': 120} {'min_samples_leaf': 70, 'min_samples_split': 160} {'max_depth': 13} {'class_weight': None, 'criterion': 'gini'}

""" #initiate random forest with the optimal parameter above, this can change depending results above"""

rfc=RandomForestClassifier(n_estimators=120,min_samples_split=160,min_samples_leaf=70,max_depth=13,criterion='gini')
rfc = rfc.fit(xtrain,ytrain)  #fit the model


#看matched在不同bracket中的fraction
#mx:得到一列xtest对应的roc的probability
mx = pd.DataFrame({"score":rfc.predict_proba(xtest)[:,1]}) 
#my:得到一列ytest
my = ytest.reset_index().drop(columns = ["index"])
#合并这两列
plotdf = mx.join(my)

"""note: the code below should be easily done using pd.cuts in pandas... 
   it's too involved now, feel free to change"""
   
#算mx是（0-0.1）（0-0.2）...中matched=1的比例
def fraction(n):
    m1 = 0
    m2 = 0
    for index, i in plotdf.iterrows():
        if i["score"] >= (n-0.1) and i["score"]<= n:
            m1 = m1 + 1
            if i["matched"] == 1:
                m2 = m2+1
    return m2/m1

list = []
for i in range(1,11):
    ii =0.1 *i
    list.append(fraction(ii)) 
 
import matplotlib.pyplot as plt
xp = [0.1, 0.2, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,1]
plt.plot(xp, list)


#%%
#可视化attribute重要性
importances = rfc.feature_importances_
std = np.std([tree.feature_importances_ for tree in rfc.estimators_], axis=0)
indices = np.argsort(importances)[::-1]# Print the feature ranking -1代表从大到小
print("Feature ranking:")
for f in range(len(xlist3)):    
    print("%2d) %-*s %f" % (f + 1, 30, x[xlist3].columns[indices[f]], importances[indices[f]]))# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(len(xlist3)), importances[indices],  color="r", yerr=std[indices], align="center")
plt.xticks(range(len(xlist3)), indices)
plt.xlim([-1, len(xlist3)])
plt.show() #图下面是attribute的索引，黑线代表的是标准差

#%%
"""apply the model prediction to all the name pairs in the entire sample"""
rfc=RandomForestClassifier(n_estimators=120,min_samples_split=160,min_samples_leaf=70,max_depth=13,criterion='gini')
rfc = rfc.fit(x,y)  #fit the model using all the data
df['pred_matched']=rfc.predict_proba(df[xlist])[:,1] #get predicted probability of y variable
