import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn import svm
import xgboost as xgb

# Loading data
train_set = pd.read_csv("dataset/train.csv")
test_set = pd.read_csv("dataset/test.csv")

# Data exploration
# print(train_set.isnull().sum())
# print(test_set.isnull().sum())
# print(train_set.head())
# print(test_set.head())
# print(train_set['toxic'].value_counts())

# Generate vocabulary
toxic_vocab = misc.sumForToxicType(train_set)

# Feature engineering
toxic_train_set = misc.feature_engineering(train_set, toxic_vocab, 'toxic', csv='dataset/toxic_train.csv')

# Modeling
# toxic_train_set, toxic_val_set = train_test_split(toxic_train_set, test_size=0.2, random_state=0)
toxic_train_X = toxic_train_set.drop(['score', 'id', 'comment_text'], axis=1).loc[:]
toxic_train_y = np.ravel(toxic_train_set.loc[:, ['score']])
# toxic_test_X = toxic_val_set.drop(['score', 'id', 'comment_text'], axis=1).loc[:]
# toxic_test_y = np.ravel(toxic_val_set.loc[:, ['score']])

# Selected model: SVM, RandomForest, GBDT, XGboost, ExtraTrees

# ////////////////////////////////////////////////////////////////
# SVM
# scale data for speeding up
# scaling = MinMaxScaler(feature_range=(-1, 1)).fit(toxic_train_X)
# toxic_train_X = scaling.transform(toxic_train_X)
# clf_svm = svm.SVC(
#     class_weight='balanced',
# )
# param_grid = {'C': [0.01, 0.1, 1, 10],
#                 'gamma': [0.001, 0.01, 0.1]}
# gs = GridSearchCV(estimator=clf_svm, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=8, verbose=5)
# gs.fit(toxic_train_X, toxic_train_y)
# logger.info("best score (roc_auc) is %s" % gs.best_score_)
# logger.info("best params are %s" % gs.best_params_)
# ////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////
# Random Forest
clf_rf = RandomForestClassifier(
    n_estimators=640,
    max_depth=15,
    min_samples_split=0.005,
    min_samples_leaf=35,
    max_features='auto',
    class_weight='balanced',
    n_jobs=8,
)
# base line model
# Accuracy : 0.9395
# AUC Score (Train): 0.869138
# CV Score : Mean - 0.8708569 | Std - 0.002699126 | Min - 0.868195 | Max - 0.8755527
# misc.modelfit(clf_rf, toxic_train_X, toxic_train_y)

# fine tune n_estimators
# optimal: 500. Actually the larger n_estimators is, the better accuracy we obatin. But given the efficiency, we
# make a trade-off here so as to select n_estimators as 500
# param_grid = {"n_estimators": np.arange(50, 500, 50)}
# gs = GridSearchCV(estimator=clf_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=8, verbose=1)
# gs.fit(toxic_train_X, toxic_train_y)
# print(gs.best_score_)
# print(gs.best_params_)
# for item in gs.grid_scores_:
#     print("mean: %s, %s" % (item[1], str(item[0])))

# fine tune max_depth and min_samples_split
# optimal: max_depth: 15, min_samples_split: 0.005
# param_grid = {"max_depth": np.arange(3, 16, 2),
#                   "min_samples_split": np.arange(0.005, 0.021, 0.005)}
# gs = GridSearchCV(estimator=clf_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=8, verbose=1)
# gs.fit(toxic_train_X, toxic_train_y)
# print(gs.best_score_)
# print(gs.best_params_)
# for item in gs.grid_scores_:
#     print("mean: %s, %s" % (item[1], str(item[0])))

# fine tune min_samples_split and min_samples_leaf
# optimal: min_samples_split:0.005, min_samples_leaf:35
# param_grid = {"min_samples_split": np.arange(0.005, 0.031, 0.005),
#                   "min_samples_leaf": np.arange(5, 51, 5)}
# gs = GridSearchCV(estimator=clf_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=8, verbose=1)
# gs.fit(toxic_train_X, toxic_train_y)
# logger.info("best score (roc_auc) is %s" % gs.best_score_)
# logger.info("best params are %s" % gs.best_params_)
# for item in gs.grid_scores_:
#     logger.info("mean: %s, %s\n" % (item[1], str(item[0])))

# fine tune max_features
# optimal: max_features: 'auto'
# feat_num = len(list(toxic_train_X.columns))
# param_grid = {"max_features": np.arange(int(np.sqrt(feat_num)), int(0.4 * feat_num), 2)}
# gs = GridSearchCV(estimator=clf_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=8, verbose=1)
# gs.fit(toxic_train_X, toxic_train_y)
# logger.info("best score (roc_auc) is %s" % gs.best_score_)
# logger.info("best params are %s" % gs.best_params_)
# for item in gs.grid_scores_:
#     logger.info("mean: %s, %s\n" % (item[1], str(item[0])))

# refine-tune n_estimators
# optimal: 640
# param_grid = {"n_estimators": np.arange(40, 801, 40)}
# gs = GridSearchCV(estimator=clf_rf, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=8, verbose=1)
# gs.fit(toxic_train_X, toxic_train_y)
# logger.info("refine-tune n_estimator given all the other parameters have been optimized and fixed")
# logger.info("best score (roc_auc) is %s" % gs.best_score_)
# logger.info("best params are %s" % gs.best_params_)
# for item in gs.grid_scores_:
#     logger.info("mean: %s, %s\n" % (item[1], str(item[0])))

# With optimization, the optimal auc is 0.8752115902, the optimal accuracy is 0.9393624092
# clf_rf.fit(toxic_train_X, toxic_train_y)
# score_auc = cross_val_score(clf_rf, toxic_train_X, toxic_train_y, cv=10, scoring='roc_auc', n_jobs=8).mean()
# score_acc = cross_val_score(clf_rf, toxic_train_X, toxic_train_y, cv=10, scoring='accuracy', n_jobs=8).mean()
# logger.info("After parameters tuning: average roc_auc is %.10f, average accuracy is %.10f" % (score_auc, score_acc))
# ////////////////////////////////////////////////////////////////

# ////////////////////////////////////////////////////////////////
# GBDT
clf_gb = GradientBoostingClassifier(
    learning_rate=0.075,
    n_estimators=800,
    max_depth=13,
    min_samples_split=0.015,
    min_samples_leaf=5,
    max_features=32,
    subsample=0.75)

# base line model
# Accuracy : 0.9409
# AUC Score (Train): 0.866622
# CV Score : Mean - 0.8708569 | Std - 0.002699126 | Min - 0.868195 | Max - 0.8755527
# misc.modelfit(clf_gb, toxic_train_X, toxic_train_y)

# fine tune n_estimators
# optimal: n_estimators: 300
# param_grid = {"n_estimators": np.arange(100, 501, 50)}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='GBDT')

# fine tune max_depth and min_samples_split
# optimal: max_depth: 13, min_samples_split: 0.015
# param_grid = {"max_depth": np.arange(3, 16, 2),
#                 "min_samples_split": np.arange(0.005, 0.021, 0.005)}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='GBDT')

# fine tune min_samples_split and min_samples_leaf
# optimal: min_samples_split: 0.015, min_samples_leaf: 5
# param_grid = {"min_samples_split": [0.01, 0.015, 0.02],
#                 "min_samples_leaf": [5, 15] + (list(np.arange(10, 71, 10)))}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='GBDT')

# fine tune max_features
# optimal: max_features: selected 32 here which is corresponding to auc score 0.879426. The reason we neglect max_depth
# as 64 is because too high value of max_features may result in overfitting
# param_grid = {"max_features": [60, 66, 72, 78, 84]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='GBDT')

# fine tune subsample
# optimal: subsample: 0.75
# param_grid = {"subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='GBDT')

# refine-tune n_estimatosr
# param_grid = {"learning_rate": [0.2],
#               "n_estimators": [300]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')
#
# param_grid = {"learning_rate": [0.1],
#               "n_estimators": [600]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')
#
# param_grid = {"learning_rate": [0.075],
#               "n_estimators": [800]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')
#
# param_grid = {"learning_rate": [0.05],
#               "n_estimators": [1200]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')
#
# param_grid = {"learning_rate": [0.04],
#               "n_estimators": [1500]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')
#
# param_grid = {"learning_rate": [0.03],
#               "n_estimators": [2000]}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_gb, param_grid, sample_weight=True, cv=5,
#                     scoring='roc_auc', n_jobs=1, method='GBDT')

# With optimization, the optimal auc is 0.8751157067, the optimal accuracy is 0.945090275
# clf_gb.fit(toxic_train_X, toxic_train_y)
# score_auc = cross_val_score(clf_gb, toxic_train_X, toxic_train_y, cv=10, scoring='roc_auc').mean()
# score_acc = cross_val_score(clf_gb, toxic_train_X, toxic_train_y, cv=10, scoring='accuracy').mean()
# logger.info("After parameters tuning: average roc_auc is %.10f, average accuracy is %.10f" % (score_auc, score_acc))
# ////////////////////////////////////////////////////////////////

# Xgboost
clf_xgb = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=307,
    max_depth=35,
    min_child_weight=1,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    nthread=8,
    scale_pos_weight=1)

# misc.modelfit_xgboost(clf_xgb, toxic_train_X, toxic_train_y)
# base line model
# Accuracy : 0.9444
# AUC Score (Train): 0.877955
# misc.modelfit(clf_gb, toxic_train_X, toxic_train_y)

# fine tune max_depth and min_child_weight(min_child_leaf)
# optimal: max_depth: 15, min_child_weight: 1
# param_grid = {"max_depth": np.arange(3, 16, 2),
#                 "min_child_weight": np.arange(1, 10, 2)}
#
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='XGBoost')

# refine-tune max_depth
# optimal: max_depth: 35
# param_grid = {"max_depth": [15, 16, 17],
#               "min_child_weight": [1, 2]}
#
#  misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='XGBoost')
#  param_grid = {"max_depth": [23, 25, 27, 29, 31, 33, 35, 37, 39]}
#  misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='XGBoost')

# fine tune gamma
# optimal: gamma: 0.1
# param_grid = {"gamma": np.arange(0, 0.5, 0.1)}
# misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
#                     scoring='roc_auc', n_jobs=8, method='XGBoost')

# fine tune subsample and colsample_bytree
if __name__ == "__main__":
    param_grid = {"subsample": np.arange(0.6, 1, 0.1),
                  "colsample_bytree": np.arange(0.6, 1, 0.1)}
    ret = misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
                              scoring='roc_auc', n_jobs=4, method='XGBoost')
    opt_subsample = ret['subsample']
    opt_colsubmple = ret['colsample_bytree']
    clf_xgb.set_params(subsample=opt_subsample, colsample_bytree=opt_colsubmple)

    param_grid = {"subsample": [opt_subsample - 0.05, opt_subsample, opt_subsample + 0.05],
                  "colsample_bytree": [opt_colsubmple - 0.05, opt_colsubmple, opt_colsubmple+ 0.05]}
    ret = misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
                        scoring='roc_auc', n_jobs=4, method='XGBoost')

    opt_subsample = ret['subsample']
    opt_colsubmple = ret['colsample_bytree']
    clf_xgb.set_params(subsample=opt_subsample, colsample_bytree=opt_colsubmple)
    param_grid = {"reg_lambda": [1e-2, 0.1, 1, 10, 100]}
    misc.run_gridsearch(toxic_train_X, toxic_train_y, clf_xgb, param_grid, sample_weight=False, cv=5,
                        scoring='roc_auc', n_jobs=8, method='XGBoost')