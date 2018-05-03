import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import misc
from sklearn.model_selection import cross_val_score
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
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

# Selected model: RandomForest, GBDT, XGboost, ExtraTrees
# linear SVM
clf_svm = svm.SVC(
         kernel='linear',
         class_weight='balanced',
         verbose=2)

# Random Forest
clf_rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=29,
    min_samples_split=0.025,
    min_samples_leaf=5,
    max_features=11,
    class_weight='balanced',
    n_jobs=4,
)

# GBDT
clf_gb = GradientBoostingClassifier(
    learning_rate=0.04,
    n_estimators=875,
    max_depth=5,
    min_samples_split=0.005,
    min_samples_leaf=5,
    max_features=23,
    subsample=0.95
)

# Xgboost
clf_xgb = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=102,
    max_depth=28,
    min_child_weight=3,
    gamma=0.4,
    subsample=0.8,
    colsample_bytree=0.6,
    reg_lambda=1.0,
    nthread=4,
    scale_pos_weight=1
)

# Extremely Randomized Trees
clf_ext = ExtraTreesClassifier(
    n_estimators=440,
    max_depth=27,
    min_samples_split=0.02,
    min_samples_leaf=5,
    max_features=33,
    class_weight='balanced',
    n_jobs=4,
    bootstrap=True,
    oob_score=True,
)

# meta_classifier as logistic regression
lr_stack = LogisticRegression(
    class_weight='balanced',
    solver='sag',
    max_iter=10000,
    n_jobs=4,
    verbose=2
)

xgb_stack = xgb.XGBClassifier(
    learning_rate=0.1,
    n_estimators=600,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    n_jobs=4
)


def train_SVM(estimator, trainX, trainY, method, n_jobs=4, skip=False):
    # SVM
    logger = misc.init_logger(method)
    xmlPath = os.path.join(os.path.dirname(__file__), "params", '%s.xml' % method.split('_')[1])
    if not skip:
        logger.info("Begin to train SVM...")
        # scale data for speeding up
        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(trainX)
        transformed_trainX = scaling.transform(trainX)
        param_grid = {'C': [1, 10, 100]}
        best_params, best_score = misc.run_gridsearch(transformed_trainX, trainY, estimator, param_grid, cv=3,
                                                      sample_weight=False,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(C=best_params['C'])
        misc.update_params_toXML(estimator, method, xmlPath)
    else:
        try:
            estimator = misc.load_params_fromXML(estimator, method, xmlPath)
        except Exception:
            return estimator
    logger.info("After parameters tuning. The current parameters are\n %s" % str(estimator.get_params()))
    return estimator


def train_RF(estimator, trainX, trainY, method, n_jobs=4, skip=False):
    # RandomForest
    logger = misc.init_logger(method)
    xmlPath = os.path.join(os.path.dirname(__file__), "params", '%s.xml' % method.split('_')[1])
    if not skip:
        logger.info("Begin to train RandomForest...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)

        # fine tune n_estimators
        param_grid = {"n_estimators": np.arange(50, 601, 50)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(n_estimators=best_params['n_estimators'])

        # fine tune max_depth and min_samples_split
        param_grid = {"max_depth": np.arange(5, 30, 2),
                      "min_samples_split": np.arange(0.005, 0.031, 0.005)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])

        # fine tune min_samples_split and min_samples_leaf
        param_grid = {"min_samples_leaf": np.arange(5, 51, 5)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(min_samples_leaf=best_params['min_samples_leaf'])

        # fine tune max_features
        feat_num = len(list(trainX.columns))
        param_grid = {"max_features": np.arange(int(np.sqrt(feat_num)), int(0.4 * feat_num), 2)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        if best_params['max_features'] == int(np.sqrt(feat_num)):
            estimator.set_params(max_features='auto')
        else:
            estimator.set_params(max_features=best_params['max_features'])

        # refine-tune n_estimators
        param_grid = {"n_estimators": np.arange(40, 1001, 40)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(n_estimators=best_params['n_estimators'])
        misc.update_params_toXML(estimator, method, xmlPath)

        logger.info("After parameters tuning, Get the CV score...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)
    else:
        try:
            estimator = misc.load_params_fromXML(estimator, method, xmlPath)
        except Exception:
            return estimator
    logger.info("After parameters tuning. The current parameters are\n %s" % str(estimator.get_params()))
    return estimator


def train_GBDT(estimator, trainX, trainY, method, n_jobs=4, skip=False):
    # GBDT
    logger = misc.init_logger(method)
    xmlPath = os.path.join(os.path.dirname(__file__), "params", '%s.xml' % method.split('_')[1])
    if not skip:
        logger.info("Begin to train GBDT...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)

        # fine tune n_estimators
        # param_grid = {"n_estimators": np.arange(50, 601, 50)}
        # best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=True, cv=5,
        #                                               scoring='roc_auc', n_jobs=n_jobs, method=method)
        # best_n_estimators = best_params['n_estimators']
        # estimator.set_params(n_estimators=best_n_estimators)
        #
        # # fine tune max_depth and min_samples_split
        # param_grid = {"max_depth": np.arange(5, 30, 2),
        #               "min_samples_split": np.arange(0.005, 0.031, 0.005)}
        # best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=True, cv=5,
        #                                               scoring='roc_auc', n_jobs=n_jobs, method=method)
        # estimator.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])
        #
        # # fine tune min_samples_split and min_samples_leaf
        # param_grid = {"min_samples_leaf": np.arange(5, 51, 5)}
        # best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=True, cv=5,
        #                                               scoring='roc_auc', n_jobs=n_jobs, method=method)
        # estimator.set_params(min_samples_leaf=best_params['min_samples_leaf'])
        #
        # # fine tune max_features
        # feat_num = len(list(trainX.columns))
        # param_grid = {"max_features": np.arange(int(np.sqrt(feat_num)), int(0.4 * feat_num), 2)}
        # best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=True, cv=5,
        #                                               scoring='roc_auc', n_jobs=n_jobs, method=method)
        # if best_params['max_features'] == int(np.sqrt(feat_num)):
        #     estimator.set_params(max_features='auto')
        # else:
        #     estimator.set_params(max_features=best_params['max_features'])
        #
        # # fine tune subsample
        # param_grid = {"subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]}
        # best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=True, cv=5,
        #                                               scoring='roc_auc', n_jobs=n_jobs, method=method)
        # estimator.set_params(subsample=best_params['subsample'])

        # refine-tune n_estimatosr
        best_n_estimators = estimator.get_params()['n_estimators']
        pairs = [(0.1,   best_n_estimators),
                 (0.075, int(best_n_estimators * 4.0 / 3)),
                 (0.05,  best_n_estimators * 2),
                 (0.04,  int(best_n_estimators * 5.0 / 2)),
                 (0.03,  int(best_n_estimators * 10.0 / 3)),
                 (0.01,  best_n_estimators * 10),
                 (0.005, best_n_estimators * 20)]
        max_n_estimators  = 2400
        opt_params = None
        opt_score = 0.0
        for learning_rate, n_estimators in pairs:
            if n_estimators > max_n_estimators:
                break
            estimator.set_params(learning_rate=learning_rate, n_estimators=n_estimators)
            auc_score, acc_score = misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)
            logger.info("With learning_rate %s, n_estimators %s, auc_score is %s, acc_score is %s" % (
            learning_rate, n_estimators, auc_score, acc_score))
            if auc_score > opt_score:
                opt_params = (learning_rate, n_estimators)
                opt_score = auc_score
        logger.info("best learning_rate is %s, best n_estimators is %s. The corresponding auc_score is %s" % (
        opt_params[0], opt_params[1], opt_score))
        estimator.set_params(learning_rate=opt_params[0], n_estimators=opt_params[1])
        misc.update_params_toXML(estimator, method, xmlPath)

        logger.info("After parameters tuning, Get the CV score...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)
    else:
        try:
            estimator = misc.load_params_fromXML(estimator, method, xmlPath)
        except Exception:
            return estimator
    logger.info("After parameters tuning. The current parameters are\n %s" % str(estimator.get_params()))
    return estimator


def train_XGB(estimator, trainX, trainY, method, n_jobs=4, skip=False):
    # Xgboost
    logger = misc.init_logger(method)
    xmlPath = os.path.join(os.path.dirname(__file__), "params", '%s.xml' % method.split('_')[1])
    if not skip:
        logger.info("Begin to train XGBoost...")
        misc.modelfit_xgboost(estimator, trainX, trainY, method, n_jobs=n_jobs)

        # fine tune max_depth and min_child_weight(min_child_leaf)
        param_grid = {"max_depth": np.arange(5, 30, 2),
                      "min_child_weight": np.arange(1, 8, 2)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(max_depth=best_params['max_depth'], min_child_weight=best_params['min_child_weight'])

        # refine-tune max_depth and min_child_weight
        opt_max_depth = best_params['max_depth']
        opt_min_child_weight = best_params['min_child_weight']
        param_grid = {"max_depth": [opt_max_depth - 1, opt_max_depth, opt_max_depth + 1],
                      "min_child_weight": [1, 2] if opt_min_child_weight == 1 else [opt_min_child_weight - 1,
                                                                                    opt_min_child_weight,
                                                                                    opt_min_child_weight + 1]}

        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(max_depth=best_params['max_depth'], min_child_weight=best_params['min_child_weight'])

        # fine tune gamma
        param_grid = {"gamma": np.arange(0, 0.5, 0.1)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(gamma=best_params['gamma'])

        # fine tune subsample and colsample_bytree
        param_grid = {"subsample": np.arange(0.6, 1.01, 0.1),
                      "colsample_bytree": np.arange(0.6, 1.01, 0.1)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        opt_subsample = best_params['subsample']
        opt_colsubsample = best_params['colsample_bytree']
        estimator.set_params(subsample=opt_subsample, colsample_bytree=opt_colsubsample)

        if abs(opt_subsample - 1.0) >= 1e-8 and abs(opt_colsubsample - 1.0) >= 1e-8:
            param_grid = {"subsample": [opt_subsample - 0.05, opt_subsample, opt_subsample + 0.05],
                          "colsample_bytree": [opt_colsubsample - 0.05, opt_colsubsample, opt_colsubsample + 0.05]}
        elif abs(opt_subsample - 1.0) >= 1e-8:
            param_grid = {"subsample": [opt_subsample - 0.05, opt_subsample, opt_subsample + 0.05],
                          "colsample_bytree": [0.95, 1.0]}
        elif abs(opt_colsubsample - 1.0) >= 1e-8:
            param_grid = {"subsample": [0.95, 1.0],
                          "colsample_bytree": [opt_colsubsample - 0.05, opt_colsubsample, opt_colsubsample + 0.05]}
        else:
            param_grid = {"subsample": [0.95, 1.0], "colsample_bytree": [0.95, 1.0]}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(subsample=best_params['subsample'], colsample_bytree=best_params['colsample_bytree'])

        # fine tune reg_lambda
        param_grid = {"reg_lambda": [1e-2, 0.1, 1, 10, 100]}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        opt_lambda = best_params['reg_lambda']
        param_grid = {
            "reg_lambda": [0, opt_lambda / 5.0, opt_lambda / 2.0, opt_lambda, opt_lambda * 2.0, opt_lambda * 5.0]}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(reg_lambda=best_params['reg_lambda'])

        # refine tune learning_rate and n_estimators
        learning_rates = [0.005, 0.01, 0.025, 0.05, 0.075, 0.1]
        opt_params = None
        opt_score = 0.0
        for learning_rate in learning_rates:
            estimator.set_params(learning_rate=learning_rate, n_estimators=5000)
            auc_score, acc_score = misc.modelfit_xgboost(estimator, trainX, trainY, method, n_jobs=n_jobs)
            best_iter = estimator.get_params()['n_estimators']
            logger.info("With learning_rate %s, n_estimators %s, auc_score is %s, acc_score is %s" % (
                learning_rate, best_iter, auc_score, acc_score))
            if auc_score > opt_score:
                opt_params = (learning_rate, best_iter)
                opt_score = auc_score
        logger.info("best learning_rate is %s, best n_estimators is %s. The corresponding auc_score is %s" % (
                    opt_params[0], opt_params[1], opt_score))
        estimator.set_params(learning_rate=opt_params[0], n_estimators=opt_params[1])
        misc.update_params_toXML(estimator, method, xmlPath)

        logger.info("After parameters tuning, Get the CV score...")
        misc.modelfit_xgboost(estimator, trainX, trainY, method, n_jobs=n_jobs)
    else:
        try:
            estimator = misc.load_params_fromXML(estimator, method, xmlPath)
        except Exception:
            return estimator
    logger.info("After parameters tuning. The current parameters are\n %s" % str(estimator.get_params()))
    return estimator


def train_EXT(estimator, trainX, trainY, method, n_jobs=4, skip=False):
    # Extremely Randomized Trees
    logger = misc.init_logger(method)
    xmlPath = os.path.join(os.path.dirname(__file__), "params", '%s.xml' % method.split('_')[1])
    if not skip:
        logger.info("Begin to train ExtraTrees...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)

        # fine tune n_estimators
        param_grid = {"n_estimators": np.arange(50, 601, 50)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(n_estimators=best_params['n_estimators'])

        # fine tune max_depth and min_samples_split
        param_grid = {"max_depth": np.arange(5, 30, 2),
                      "min_samples_split": np.arange(0.005, 0.031, 0.005)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'])

        # fine tune min_samples_split and min_samples_leaf
        param_grid = {"min_samples_leaf": np.arange(5, 51, 5)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(min_samples_leaf=best_params['min_samples_leaf'])

        # fine tune max_features
        feat_num = len(list(trainX.columns))
        param_grid = {"max_features": np.arange(int(np.sqrt(feat_num)), int(0.4 * feat_num), 2)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        if best_params['max_features'] == int(np.sqrt(feat_num)):
            estimator.set_params(max_features='auto')
        else:
            estimator.set_params(max_features=best_params['max_features'])

        # refine-tune n_estimators
        param_grid = {"n_estimators": np.arange(40, 1001, 40)}
        best_params, best_score = misc.run_gridsearch(trainX, trainY, estimator, param_grid, sample_weight=False, cv=5,
                                                      scoring='roc_auc', n_jobs=n_jobs, method=method)
        estimator.set_params(n_estimators=best_params['n_estimators'])
        misc.update_params_toXML(estimator, method, xmlPath)

        logger.info("After parameters tuning, Get the CV score...")
        misc.modelfit(estimator, trainX, trainY, method, n_jobs=n_jobs)
    else:
        try:
            estimator = misc.load_params_fromXML(estimator, method, xmlPath)
        except Exception:
            return estimator
    logger.info("After parameters tuning. The current parameters are\n %s" % str(estimator.get_params()))
    return estimator


def run_ensemble(clf_rf, clf_gb, clf_xgb, clf_ext, trainX, trainY, method, n_jobs=4, skip_cv=False):
    # Ensemble
    logger = misc.init_logger(method)
    logger.info("Begin to Ensemble...")
    clf_vote_soft = VotingClassifier(
        estimators=[
            ('rf', clf_rf),
            ('gbdt', clf_gb),
            ('xgboost', clf_xgb),
            ('extraTree', clf_ext)
        ],
        weights=[2, 1, 2, 1],
        voting='soft'
    )

    # sclf = StackingClassifier(classifiers=[clf_rf, clf_gb, clf_xgb, clf_ext],
    #                           meta_classifier=lr_stack,
    #                           verbose=1)
    #
    # sclf_prob = StackingClassifier(classifiers=[clf_rf, clf_gb, clf_xgb, clf_ext],
    #                                use_probas=True,
    #                                average_probas=False,
    #                                meta_classifier=lr_stack,
    #                                verbose=1)
    #
    # sclf_xgb = StackingClassifier(classifiers=[clf_rf, clf_gb, clf_xgb, clf_ext],
    #                               meta_classifier=xgb_stack,
    #                               verbose=1)

    # logger.info("Begin to compare CV scores between different classifiers when ensembling...")
    # for clf, label in zip([clf_rf, clf_gb, clf_xgb, clf_ext, clf_vote_soft],
    #                       ['Random Forest',
    #                        'GBDT',
    #                        'XGBoost',
    #                        'ExtraTrees',
    #                        'SoftVotingClassifier']):
    #     estimator, auc_score, acc_score = misc.modelfit(clf, trainX, trainY, method, n_jobs=n_jobs)
    #     if label == 'SoftVotingClassifier':
    #         classifier = clf_vote_soft
    #     logger.info(
    #         'Using %s as meta classifier, average roc_auc is %.10f, average accuracy is %.10f' % (
    #             label, auc_score, acc_score))


    clf_vote_soft.fit(trainX, trainY)
    if not skip_cv:
        score_auc = cross_val_score(clf_vote_soft, trainX, trainY, cv=5, scoring='roc_auc', verbose=1,
                                    n_jobs=n_jobs).mean()
        score_acc = cross_val_score(clf_vote_soft, trainX, trainY, cv=5, scoring='accuracy', verbose=1,
                                    n_jobs=n_jobs).mean()
        logger.info('Using SoftVotingClassifier as meta classifier, average roc_auc is %.10f, average accuracy is %.10f' % (
            score_auc, score_acc))

    joblib.dump(clf_vote_soft, 'models/%s_lr_soft_vote.model' % method)
    return clf_vote_soft


def train(train_set, comment_type, vocab, **skip_clf):
    global clf_rf, clf_gb, clf_xgb, clf_ext
    print("Start to train comment type: %s" % comment_type)

    # Feature engineering
    train_set = misc.feature_engineering(train_set, vocab, comment_type)

    # Modeling
    # train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=0)
    trainX = train_set.drop(['score', 'id', 'comment_text'], axis=1).loc[:]
    trainY = np.ravel(train_set.loc[:, ['score']])
    # testX = val_set.drop(['score', 'id', 'comment_text'], axis=1).loc[:]
    # testY = np.ravel(val_set.loc[:, ['score']])

    # clf_svm = train_SVM(clf_svm, trainX, trainY, 'SVM_%s' % comment_type)
    skip_RF = True if ('skip_RF' in skip_clf and skip_clf['skip_RF'] is True) else False
    skip_GBDT = True if ('skip_GBDT' in skip_clf and skip_clf['skip_GBDT'] is True) else False
    skip_XGB = True if ('skip_XGB' in skip_clf and skip_clf['skip_XGB'] is True) else False
    skip_ExtTree = True if ('skip_ExtTree' in skip_clf and skip_clf['skip_ExtTree'] is True) else False
    skip_CV = True if ('skip_CV' in skip_clf and skip_clf['skip_CV'] is True) else False

    clf_rf = train_RF(clf_rf, trainX, trainY, 'RandomForest_%s' % comment_type, skip=skip_RF)
    clf_gb = train_GBDT(clf_gb, trainX, trainY, 'GBDT_%s' % comment_type, skip=skip_GBDT)
    clf_xgb = train_XGB(clf_xgb, trainX, trainY, 'XGBoost_%s' % comment_type, skip=skip_XGB)
    clf_ext = train_EXT(clf_ext, trainX, trainY, 'ExtraTree_%s' % comment_type, skip=skip_ExtTree)

    clf_vote_soft = run_ensemble(clf_rf, clf_gb, clf_xgb, clf_ext, trainX, trainY, 'Ensemble_%s' % comment_type, skip_cv=skip_CV)
    return clf_vote_soft


def predict(test_set, comment_type, vocab, estimator, use_proba=False):
    # Feature engineering
    test_set = misc.feature_engineering(test_set, vocab, comment_type, is_test=True)

    df_test = test_set.drop(['id', 'comment_text'], axis=1).loc[:]
    if use_proba:
        result = estimator.predict_proba(df_test)
    else:
        result = estimator.predict(df_test)
    return result[:, 1]


if __name__ == "__main__":
    comment_type = os.path.splitext(os.path.basename(__file__))[0]

    # Generate vocabulary
    vocab = misc.sumForToxicType(train_set)

    clf = train(train_set, comment_type, vocab, skip_RF=True, skip_GBDT=True)
    result = predict(test_set, comment_type, vocab, clf, use_proba=True)
