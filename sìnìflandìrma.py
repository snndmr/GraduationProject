import os
import numpy as np
import pandas as pd
import seaborn as sns
from nltk import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, \
    GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.linear_model as skl_lm
from sklearn.utils import class_weight
import pickle
#!pip install catboost
from catboost import CatBoostRegressor

# !pip install lightgbm
# conda install -c conda-forge lightgbm
from lightgbm import LGBMRegressor, LGBMClassifier

# !pip install xgboost
from sklearn.utils import class_weight
from xgboost import XGBRegressor

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

# VERININ HAZIRLANMASI

cur_dir = os.getcwd()
cur_dir
#os.chdir(r'C:\Users\Public\graduationProject')

# Kaydedilmiş Verinin Çağırılması
#Veri ön işleme ile bir daha uğraşmamak için
train_df = pd.read_pickle("datasets/liftUp_cnc/prepared_data/train_df.pkl")
test_df = pd.read_pickle("datasets/liftUp_cnc/prepared_data/test_df.pkl")

all_data = [train_df, test_df]
drop_list = ["datetime"]

for data in all_data:
    data.drop(drop_list, axis=1, inplace=True)

train_df.head()

# Data Preprocessing

train_df.isnull().sum()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.04)
    quartile3 = dataframe[variable].quantile(0.96)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < low_limit) | (dataframe[variable] > up_limit)].any(axis=None):
        print(variable, "yes")


outlier_thresholds(train_df, "age")

has_outliers(train_df, "age")

outlier_col = train_df[['voltmean','rotatemean','pressuremean','vibrationmean','voltsd','rotatesd','pressuresd','vibrationsd','age']]

for col in outlier_col:
    has_outliers(train_df, col)

train_df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

outlier_thresholds(train_df, "rotatemean")
outlier_thresholds(train_df, "pressuremean")
outlier_thresholds(train_df, "vibrationmean")
outlier_thresholds(train_df, "pressuresd")

# Modeling

# Lojistik Regresyon (Logistic Regression)

y = train_df["failure"]
X = train_df.drop(["failure"], axis=1)

X.head()
y.head()

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

log_model.predict(X)[0:10]
y[0:10]

log_model.predict_proba(X)[0:10] #olasılıkları gösterir 0 gerçekleşme 1 gerçekleşme olasılığını temsil eder indexler 0.5 e hangisi yakınsa onu tahminler
y_pred = log_model.predict(X) #LR ile tahmin edilen değerleri y_pred e kaydediyorum
accuracy_score(y, y_pred) # y gerçek değerler y_pred tahmin edilen değerler

cross_val_score(log_model, X, y, cv=10).mean() #veriyi 10 a bölelim 9 uyla eğitelim 1 iyle test edelim ve hata ortalamamıza bakalım.

print(classification_report(y, y_pred))



logit_roc_auc = roc_auc_score(y, log_model.predict(X))
fpr, tpr, thresholds = roc_curve(y, log_model.predict_proba(X)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label='AUC (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

# RF

rf_model = RandomForestClassifier(random_state=12345).fit(X, y)

cross_val_score(rf_model, X, y, cv=10).mean()

rf_params = {"max_depth": [3,5, 8, None],
             "max_features": [3, 5, 7],
             "n_estimators": [100,200, 500,1000],
             "min_samples_split": [2, 5, 10,30]}

rf_model = RandomForestClassifier(random_state=12345)

gs_cv = GridSearchCV(rf_model,
                     rf_params,
                     cv=10,
                     n_jobs=-1,
                     verbose=2).fit(X, y)

gs_cv.best_params_

#class_weights = class_weight.compute_class_weight("balanced",np.unique(y), y) #y-hedef değişken

rf_tuned = RandomForestClassifier(**gs_cv.best_params_,class_weight={1:1, 0:20})
cross_val_score(rf_tuned, X, y, cv=10).mean()

# Feature Importance : NA olanlar çöp. Değişkenlerin önem düzeyini anlıyoruz.

os.getcwd()
os.chdir(r'C:\Users\Public\graduationProject')

rf_tuned.feature_importances_
Importance = pd.DataFrame({'Importance': rf_tuned.feature_importances_ * 100,
                           'Feature': X.columns})

plt.figure(figsize=(10, 30))
sns.barplot(x="Importance", y="Feature", data=Importance.sort_values(by="Importance", ascending=False))
plt.title('Feature Importance ')
plt.show()
plt.savefig('rf_importance.png')

# TUM MODELLER CV YONTEMI - Raporda kullanabilirsin..

models = [('LR', LogisticRegression()),
          ('KNN', KNeighborsClassifier()),
          ('CART', DecisionTreeClassifier()),
          ('RF', RandomForestClassifier()),
          ('SVM', SVC(gamma='auto')),
          ('XGB', GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=10, random_state=123456)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure(figsize=(15, 10))
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#RF için,

# TUM MODELLER CV YONTEMI - Raporda kullanabilirsin..

models = [('RF', rf_tuned)]

# evaluate each model in turn
result_rf = []
name_rf = []


for name, model in models:
    model.fit(X,y)
    kfold = KFold(n_splits=10, random_state=12345)
    cv_results = cross_val_score(model, X, y, cv=10, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)




# Tune Edilmiş Modellerin Kaydedilmesi
# models isminde bir klasör açıyorum.
# working directory'i kaydediyorum.
cur_dir = os.getcwd()
cur_dir

# directory'yi değiştiriyorum:
os.chdir('projects/liftUp_cnc/models')

#pickle ile diske iniyoruz. ramden diske iniyoruz. Dataframe,model nesnesi kaydetmede kullanılır.
#model[0] Knn , model[1] modelin kendisi var. KNN.pkl model nesnesidir.
for model in models:
    pickle.dump(model[1], open(str(model[0]) + ".pkl", 'wb'))

# kaydedilmiş bir modeli cagiralim: Öğrenilmiş modeli çağırma işlemidir. 5 gözlemi tahmin et dedik
#atıyorum bunu birisine böyle kullanıyor.
rf = pickle.load(open('RF.pkl', 'rb'))
rf.predict(X)[0:15]
y[0:15]

