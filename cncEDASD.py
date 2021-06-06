# 1. GENEL RESIM

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing
import missingno as msno
import os
import sklearn
from sklearn.metrics import accuracy_score
import warnings

#cur_dir = os.getcwd()
#cur_dir
#os.chdir(r'C:\Users\Public\graduationProject')

pd.pandas.set_option('display.max_columns', None)

# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("datasets/cnc_outsd.csv")
test = pd.read_csv("datasets/cnc_test_outsd.csv")
df = train.append(test).reset_index()


df = pd.read_csv("datasets/cnc_outsd.csv")

failure_label = preprocessing.LabelEncoder()
df["failure"] = failure_label.fit_transform(df["failure"])

failure_label = preprocessing.LabelEncoder()
df["model"] = failure_label.fit_transform(df["model"])



df.head()
df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()

# 2. KATEGORIK DEGISKEN ANALIZI



df.failure.unique()
df.failure.value_counts()

df["failure"].value_counts() * 100 / len(df)
df["age"].value_counts() #buu


# Kac kategorik değişken var ve isimleri neler?
cat_cols = [col for col in df.columns if df[col].dtypes == 'O' and len(df[col].unique()) < 3]
print('Kategorik Değişken Sayısı: ', len(cat_cols))
print(cat_cols)

def cats_summary(data, categorical_cols, number_of_classes=10):
    var_count = 0  # Kaç kategorik değişken olduğu raporlanacak
    vars_more_classes = []  # Belirli bir sayıdan daha fazla sayıda sınıfı olan değişkenler saklanacak.
    for var in data:
        if var in categorical_cols:
            if len(list(data[var].unique())) <= number_of_classes:  # sınıf sayısına göre seç
                print(pd.DataFrame({var: data[var].value_counts(),
                                    "Ratio": 100 * data[var].value_counts() / len(data)}),
                      end="\n\n\n")
                var_count += 1
            else:
                vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)


cats_summary(df, cat_cols)

# 3. SAYISAL DEGISKEN ANALIZI

# Sayısal değişkenlere genel bakış:
df.describe().T
df.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T

# Veri setinde kaç sayısal değişken var?
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and
            col not in "failure" and col not in "machineID"]
print(num_cols)


# Sayısal değişkenlerin hepsini otomatik olarak nasıl analiz ederiz?
def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

# 4. TARGET/DEPENDENT/OUTPUT/ ANALIZI

# Failure değişkeninin dağılımını inceleyelim
df["failure"].value_counts()

cat_cols

# KATEGORIK DEGISKENLERE GORE TARGET ANALIZI
# Nasıl yani? Kategorik değişkenlere göre grup by yapıp failure'a göre ortalamasını alarak.
df.groupby("model")["failure"].mean()

# Peki bunu tüm değişkenlere otomatik olarak nasıl yapabiliriz?
def target_summary_with_cat(data, target):
    cats_names = [col for col in data.columns if df[col].dtypes == 'O' and len(data[col].unique()) < 3 and col not in target]
    for var in cats_names:
        print(pd.DataFrame({"TARGET_MEAN": data.groupby(var)[target].mean()}), end="\n\n\n")


target_summary_with_cat(df, "failure")


# SAYISAL DEGISKENLERE GORE TARGET ANALIZI
def target_summary_with_nums(data, target):
    num_names = [col for col in data.columns if len(data[col].unique()) > 5
                 and df[col].dtypes != 'O'
                 and col not in target
                 and col not in "machineID"]

    for var in num_names:
        print(df.groupby(target).agg({var: np.mean}), end="\n\n\n")


target_summary_with_nums(df, "failure")

# 5.SAYISAL DEGISKENLERIN BIRBIRLERINE GORE INCELENMESI

sns.scatterplot(x="vibration", y="volt", data=df)
plt.show()

sns.lmplot(x="vibration", y="volt", data=df)
plt.show()


df.corr()


## FEATURE ENGINEERING

# DATA PRE-PROCESSING & FEATURE ENGINEERING


# 1. AYKIRI DEGER ANALIZI
# 2. EKSIK DEGER ANALIZI
# 3. LABEL ENCODING
# 4. ONE-HOT ENCODING
# 5. SUPER-CATEGORY CATCHING
# 6. RARE ENCODING
# 7. STANDARDIZATION
# 8. FEATURE ENGINEERING
# 9. RECAP

# EKSİK DEĞER ANALİZİ

def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na

cols_with_na = missing_values_table(df)
#Eksik değerimiz yok

# 8. FEATURE ENGINEERING

# NUMERIC TO CATEGORICAL: Bu da bir yöntemdir. Buna daha sonra one-hot uygula 1-0 yazsın
df.loc[(df['age'] >= 0) & (df['age'] < 6), 'NEW_AGE_CAT'] = 'yeni'
df.loc[(df['age'] >= 6) & (df['age'] <= 15), 'NEW_AGE_CAT'] = 'orta'
df.loc[(df['age'] >= 16), 'NEW_AGE_CAT'] = 'eski'

print(df.tail())

#os.getcwd()
#os.chdir(r'C:\Users\Public\graduationProject')

df.groupby("model").agg({"failure": "mean", "age": ["count", "mean"]})
df.columns
df.groupby("machineID").agg({"model": ["count", "mean","min","max"], "age": ["count", "mean","min","max"]})
df.groupby("model").agg({"machineID": "mean", "age": ["count", "mean"]})


#yani işlemin başarılı olması makine yılının küçük olmasına da bağlı 10 ya da 7 den daha küçükler yüksek ihtimalle daha başarılı
df.groupby("failure").agg({"age":["count","mean"]})

# error countlar için yeni değişken
df.loc[((df['error1count'] + df['error2count'] + df['error3count'] + df['error4count'] + df['error5count']) >= 5), "NEW_IS_RISKY"] = "HIGH RISK" #2
df.loc[((df['error1count'] + df['error2count'] + df['error3count'] + df['error4count'] + df['error5count']) >= 1), "NEW_IS_RISKY"] = "RISK" #1
df.loc[((df['error1count'] + df['error2count'] + df['error3count'] + df['error4count'] + df['error5count']) == 0), "NEW_IS_RISKY"] = "LOW RISK" #0
df.head(25)


failure_label = preprocessing.LabelEncoder()
df["NEW_AGE_CAT"] = failure_label.fit_transform(df["NEW_AGE_CAT"])

failure_label = preprocessing.LabelEncoder()
df["NEW_IS_RISKY"] = failure_label.fit_transform(df["NEW_IS_RISKY"])

# Veriyi kaydetme

train_df = df[df['failure'].notnull()]
test_df = df[df['failure'].isnull()]

train_df.to_pickle("datasets/liftUp_cnc/prepared_data/train_df_sd.pkl")
test_df.to_pickle("datasets/liftUp_cnc/prepared_data/test_df_sd.pkl")


df.head(125)