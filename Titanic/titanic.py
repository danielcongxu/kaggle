import warnings
import pandas as pd
import numpy as np
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