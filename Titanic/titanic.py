import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from misc import show_confusion_matrix
from misc import SKlearnHelper
from scipy import stats
import sklearn as sk
import itertools
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
import xgboost as xgb
import lightgbm as lgb
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings("ignore")
sns.set(style="white", context="notebook", palette="deep")

train_set = pd.read_csv("data set/train.csv")
test_set = pd.read_csv("data set/test.csv")
combine_set = pd.concat([train_set.drop('Survived', axis=1), test_set])

surv = train_set[train_set["Survived"] == 1]
nosurv = train_set[train_set["Survived"] == 0]
surv_col = "blue"
nosurv_col = "red"

# Inspect each feature by plot
# plt.figure(figsize=[12, 10])
# plt.subplot(331)
# # The age column ranges from 0 - 80. We can inspect it from pd.describe()
# sns.distplot(surv["Age"].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
# sns.distplot(nosurv["Age"].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col, axlabel="Age")
# plt.subplot(332)
# sns.barplot("Sex", "Survived", data=train_set)
# plt.subplot(333)
# sns.barplot("Pclass", "Survived", data=train_set)
# plt.subplot(334)
# sns.barplot("Embarked", "Survived", data=train_set)
# plt.subplot(335)
# sns.barplot("SibSp", "Survived", data=train_set)
# plt.subplot(336)
# sns.barplot("Parch", "Survived", data=train_set)
# plt.subplot(337)
# sns.distplot(np.log(surv["Fare"].dropna().values + 1), kde=False, color=surv_col)
# sns.distplot(np.log(nosurv["Fare"].dropna().values + 1), kde=False, color=nosurv_col)
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.35)
# plt.show()

# Fill missing values
train_set['Embarked'].iloc[61] = "C"
train_set['Embarked'].iloc[829] = "C"
test_set['Fare'].iloc[152] = combine_set['Fare'][combine_set['Pclass'] == 3].dropna().median()

# Feature engineering
combine_set = pd.concat([train_set.drop('Survived', axis=1), test_set])
combine_set['Child'] = combine_set['Age'] <= 10
combine_set['Cabin_known'] = combine_set['Cabin'].isnull() == False
combine_set['Age_known'] = combine_set['Age'].isnull() == False
combine_set['Family'] = combine_set['SibSp'] + combine_set['Parch']
combine_set['Alone'] = (combine_set['SibSp'] + combine_set['Parch']) == 0
combine_set['Large_Family'] = (combine_set['SibSp'] > 2) | (combine_set['Parch'] > 3)
combine_set['Deck'] = combine_set['Cabin'].str[0]
combine_set['Deck'] = combine_set['Deck'].fillna(value='U')
combine_set['Ttype'] = combine_set['Ticket'].str[0]
combine_set['Title'] = combine_set['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
combine_set['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine_set['Fare'] + 1))).astype('int')
combine_set['Bad_Ticket'] = combine_set['Ttype'].isin(['3', '4', '5', '6', '7', '8', 'A', 'L', 'W'])
combine_set['Young'] = (combine_set['Age'] <= 30) | combine_set['Title'].isin(['Master', 'Miss', 'Mlle'])
combine_set['Shared_ticket'] = np.where(combine_set.groupby('Ticket')['Name'].transform('count') > 1, 1, 0)
combine_set['Ticket_group'] = combine_set.groupby('Ticket')['Name'].transform('count')
combine_set['Fare_eff'] = combine_set['Fare'] / combine_set['Ticket_group']
combine_set['Fare_eff_cat'] = np.where(combine_set['Fare_eff'] > 16.0, 2, 1)
combine_set['Fare_eff_cat'] = np.where(combine_set['Fare_eff'] < 8.5, 0, combine_set['Fare_eff_cat'])

survived = train_set['Survived']
test_set = combine_set.iloc[len(train_set):]
train_set = combine_set.iloc[:len(train_set)]
train_set['Survived'] = survived
surv1 = train_set[train_set['Survived'] == 1]
nosurv1 = train_set[train_set['Survived'] == 0]

# Process child
# Conclusion: Male children appear to have a survival advantage in 2nd and 3rd class.
# We should include the Child feature in our model testing.
sns.factorplot(x='Sex', y="Survived", hue="Child", col="Pclass", data=train_set, aspect=0.9, size=3.5, ci=95.0)
tab_class = pd.crosstab(train_set['Child'], train_set['Pclass'])
tab_sex = pd.crosstab(train_set['Child'], train_set['Sex'])
print(tab_class)
print(tab_sex)

# Process cabin known
# Conclusion: There remains a potential trend for males and for 3rd class passengers but the uncertainties are large.
# This feature should be tested in the modelling stage.
sns.factorplot(x='Sex', y="Survived", hue="Cabin_known", col="Pclass", data=train_set, aspect=0.9, size=3.5, ci=95.0)
tab_cabin = pd.crosstab(train_set['Cabin_known'], train_set['Survived'])
print(tab_cabin)


# Process Deck
# Conclusion: the best decks for survival were B, D, and E with about 66% chance
tab_deck = pd.crosstab(train_set['Deck'], train_set['Survived'])
print(tab_deck)
sns.factorplot(x="Deck", y="Survived", data=train_set, kind="bar", color="blue")

# Ttype and Bad_ticket
# Conclusion: Based on this plot we define a new feature called Bad_ticket under which
# we collect all the ticket numbers that start with digits which suggest less than 25% survival (e.g. 4, 5, or A)
# Bad_ticket might be a lower order effect that could give us some additional accuracy.
# We should test it out in the modelling stage.
tab_ttype = pd.crosstab(train_set['Ttype'], train_set['Survived'])
print(tab_ttype)
sns.factorplot(x="Ttype", y="Survived", data=train_set, kind="bar", color="blue")

# Process Age known
# Conclusion: There is a strong impact of Sex and Pclass on this new feature.
# This might be enough to explain all the variance in the Age_known variable. We should test the predictive power in our modelling.
tab_ageknown = pd.crosstab(train_set["Age_known"], train_set['Survived'])
print(tab_ageknown)
sns.factorplot(x="Age_known", y="Survived", data=train_set, kind="bar", color="red")

# Process family
# Conclusion: Again, we find that having 1-3 family members works best for survival.
tab_family = pd.crosstab(train_set['Family'], train_set['Survived'])
tab_family.div(tab_family.sum(1), axis=0).plot(kind="bar", stacked=True)
plt.xlabel("Family members")
plt.ylabel("Survival percentage")

# Proces Alone
# Conclusion: Travelling alone appears bad enough
tab_alone = pd.crosstab(train_set["Alone"], train_set['Survived'])
print(tab_alone)
sns.factorplot(x="Alone", y="Survived", data=train_set, kind="bar", color="blue")

# Process larger family
# Conclusion: having a large family appears to be not good for survival
tab_large_family = pd.crosstab(train_set["Large_Family"], train_set["Survived"])
print(tab_large_family)
sns.factorplot(x="Large_Family", y="Survived", data=train_set, kind="bar", color="blue")

# Process shared ticket
# Conclusion: Sharing a ticket appears to be good for survival.
tab_shared_ticket = pd.crosstab(train_set['Shared_ticket'], train_set['Survived'])
print(tab_shared_ticket)
sns.factorplot(x="Shared_ticket", y="Survived", data=train_set, kind="bar", color="blue")

# Process Title
# Conclusion:
print(combine_set['Age'].groupby(combine_set['Title']).count())
print(combine_set['Age'].groupby(combine_set['Title']).mean())
dummy = combine_set[combine_set['Title'].isin(['Mr','Miss','Mrs','Master'])]
dummy['Age'].hist(by=dummy['Title'], bins=np.arange(0, 81, 1))

plt.figure(figsize=(6, 6))
tab_young = pd.crosstab(train_set['Young'], train_set['Survived'])
print(tab_young)
sns.barplot(x="Young", y="Survived", data=train_set)

# Process Fare_cat (Fare_category)
tab_fare_cat = pd.crosstab(train_set['Fare_cat'], train_set['Survived'])
print(tab_fare_cat)
plt.figure(figsize=(6, 6))
sns.barplot(x="Fare_cat", y="Survived", data=train_set)

# Preparation for modeling
combine_set = pd.concat([train_set.drop("Survived", axis=1), test_set])
survived = train_set['Survived']
combine_set['Sex'] = combine_set['Sex'].astype("category")
combine_set['Sex'].cat.categories = [0, 1]
combine_set['Sex'] = combine_set['Sex'].astype("int")
combine_set['Embarked'] = combine_set['Embarked'].astype("category")
combine_set['Embarked'].cat.categories = [0, 1, 2]
combine_set['Embarked'] = combine_set['Embarked'].astype("int")
combine_set['Deck'] = combine_set['Deck'].astype("category")
combine_set['Deck'].cat.categories = [0, 1, 2, 3, 4, 5, 6, 7, 8]
combine_set['Deck'] = combine_set['Deck'].astype("int")
train_set = combine_set.iloc[:len(train_set)]
test_set = combine_set.iloc[len(train_set):]
train_set['Survived'] = survived

plt.subplots( figsize=( 12 , 10 ) )
sns.heatmap(train_set.drop('PassengerId', axis=1).corr(), vmax=1.0, square=True, annot=True)

######################################################################
# Modeling
train_set, val_set = train_test_split(train_set, test_size=0.2, random_state=0)
cols = ['Sex', 'Pclass', 'Cabin_known', 'Large_Family', 'Shared_ticket', 'Young', 'Alone', 'Child']
tcols = np.append(['Survived'], cols)
df = train_set.loc[:, tcols].dropna()
X = df.loc[:, cols]
y = np.ravel(df.loc[:,['Survived']]) # ravel is equal to flatten
df_test = val_set.loc[:, tcols].dropna()
X_test = df_test.loc[:, cols]
y_test = np.ravel(df_test.loc[:, ['Survived']])

# Logistic Regression
classifier_lgr = LogisticRegression()
classifier_lgr.fit(X, y)
score_lgr = cross_val_score(classifier_lgr, X, y, cv=5).mean()
print("The mean evaluation accuracy of logistic regression is %s" % score_lgr)

# Perceptron
classifier_pctr = Perceptron(class_weight="balanced")
classifier_pctr.fit(X, y)
score_pctr = cross_val_score(classifier_pctr, X, y, cv=5).mean()
print("The mean evaluation accuracy of perceptron is %s" % score_pctr)

# KNN
classifier_knn = KNeighborsClassifier(n_neighbors=len(X.columns), weights='distance')
classifier_knn.fit(X, y)
score_knn = cross_val_score(classifier_knn, X, y, cv=5).mean()
print("The mean evaluation accuracy of KNN is %s" % score_knn)

# SVM
classifier_svm = svm.SVC(class_weight="balanced")
classifier_svm.fit(X, y)
score_svm = cross_val_score(classifier_svm, X, y, cv=5).mean()
print("The mean evaluation accuracy of SVM is %s" % score_svm)

# Decision Tree
# http://blog.csdn.net/li980828298/article/details/51172744
classifier_dtree = tree.DecisionTreeClassifier(class_weight="balanced", min_weight_fraction_leaf=0.01)
classifier_dtree.fit(X, y)
score_dtree = cross_val_score(classifier_dtree, X, y, cv=5).mean()
print("The mean evaluation accuracy of SVM is %s" % score_dtree)

# Tuning parameters, illustrated with SVM
# Grid search
clf_svm = svm.SVC(class_weight="balanced")
param_grid = {"C": [0.1, 0.3, 1.0, 3.0, 10.0],
              "kernel": ["linear", "rbf", "poly", "sigmoid"]}
gs = GridSearchCV(estimator=clf_svm, param_grid=param_grid, scoring="accuracy", cv=3)
gs.fit(X, y)
print("best score: %f" % gs.best_score_)
print(gs.best_params_)

class_names = ["Dead", "Alive"]
cnf_mat = confusion_matrix(y_pred=classifier_svm.predict(X_test), y_true=y_test)
show_confusion_matrix(cnf_mat, class_names)

# Model evaluation
# cross validation - cross_val_score on training set

# Ranking models
models = pd.DataFrame({'Model': ["Support Vector Machine", "KNN", "Logistic Regression",
                                 "Decision Tree", "Perceptron"],
                       'Score': [score_svm, score_knn, score_lgr, score_dtree, score_pctr]})
models.sort_values(by='Score', axis=0, ascending=False)

# Ensemble methods
classifier_vote = VotingClassifier(estimators=[
    ('knn', classifier_knn),
    ('svm', classifier_svm),
    ("logistic", classifier_lgr),
    ("decisiontree", classifier_dtree),
    ("perceptron", classifier_pctr)],
    weights=[2, 3, 2, 3, 1],
    voting='hard'
)
classifier_vote.fit(X, y)
score_votes = cross_val_score(classifier_vote, X, y, cv=5, scoring='accuracy')
print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (score_votes.mean(), score_votes.std()))

train = X
test = test_set
n_train = train.shape[0]
n_test = test.shape[0]
SEED = 0
NFOLDS = 5
kf = KFold(n_splits=NFOLDS, random_state=SEED)


def get_oof(clf, x_train, y_train, x_test):
    oof_train = np.zeros((n_train,))
    oof_test = np.zeros((n_test,))
    oof_test_skf = np.empty((NFOLDS, n_test))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)
        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = np.median(oof_test_skf, axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


lgr_params = {
    'tol': 1e-4,
    "C": 1.0}

pctr_params = {
    'alpha': 1e-4,}

knn_params = {
    'n_neighbors': 10,
    'weights':  'distance'}

svm_params = {
    "C": 1.0,
    "kernel": "rbf"}

dtree_params = {
    "min_weight_fraction_leaf": 0.01,
    "criterion": "gini"}

lgr = SKlearnHelper(clf=LogisticRegression, seed=SEED, params=lgr_params)
pctr = SKlearnHelper(clf=Perceptron, seed=SEED, params=pctr_params)
knn = SKlearnHelper(clf=KNeighborsClassifier, seed=SEED, params=knn_params)
svm = SKlearnHelper(clf=svm.SVC, seed=SEED, params=svm_params)
dtree = SKlearnHelper(clf=tree.DecisionTreeClassifier, seed=SEED, params=dtree_params)

y_train = y
train = X
foo = test.loc[:, cols]
x_train = train.values
x_test = foo.values

lgr_oof_train, lgr_oof_test = get_oof(lgr, x_train, y_train, x_test)
pctr_oof_train, pctr_oof_test = get_oof(pctr, x_train, y_train, x_test)
knn_oof_train, knn_oof_test = get_oof(knn, x_train, y_train, x_test)
svm_oof_train, svm_oof_test = get_oof(svm, x_train, y_train, x_test)
dtree_oof_train, dtree_oof_test = get_oof(dtree, x_train, y_train, x_test)

print("Training is complete")

base_predictions_train = pd.DataFrame(
    {"LogisticRegression": lgr_oof_train.ravel(),
     "Perceptron": pctr_oof_train.ravel(),
     "KNN": knn_oof_train.ravel(),
     "SVM": svm_oof_train.ravel(),
     "DecisionTree": dtree_oof_train.ravel()})

print(base_predictions_train.head())

# Stacking of classifiers that have less correlation gives better results.
# Observe based on heatmap (covariance matrix)
plt.figure(figsize=(12, 10))
sns.heatmap(base_predictions_train.corr(), vmax=1.0, square=True, annot=True)
plt.show()

x_train = np.concatenate((lgr_oof_train, pctr_oof_train, knn_oof_train, svm_oof_train, dtree_oof_train), axis=1)
x_test = np.concatenate((lgr_oof_test, pctr_oof_test, knn_oof_test, svm_oof_test, dtree_oof_test), axis=1)

clf_stack = xgb.XGBClassifier(
    #learning_rate = 0.02,
    n_estimators= 2000,
    max_depth= 4,
    min_child_weight= 2,
    #gamma=1,
    gamma=0.9,
    subsample=0.8,
    colsample_bytree=0.8,
    objective= 'binary:logistic',
    scale_pos_weight=1)
clf_stack = clf_stack.fit(x_train, y_train)
stack_pred = clf_stack.predict(x_test)

scores = cross_val_score(clf_stack, x_train, y_train, cv=5)
print("Mean score = %.3f, Std deviation = %.3f" % (np.mean(scores), np.std(scores)))

df2 = test.loc[:, cols].fillna(method='pad')
surv_pred = classifier_vote.predict(df2)

submit = np.c_[test.loc[:, "PassengerId"], surv_pred]
np.savetxt("titanic.csv",
           submit,
           delimiter=',',
           header='PassengerId,Survived',
           comments='',
           fmt="%d")