%matplotlib inline
import matplotlib.pylab as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn import preprocessing

from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax

from dmba import regressionSummary, exhaustive_search
from dmba import backward_elimination, forward_selection, stepwise_selection
from dmba import adjusted_r2_score, AIC_score, BIC_score

from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


house_df = pd.read_csv('/Users/houetsu/Downloads/Python/Kaggle/Houe_Price/Original_Data/train.csv', index_col='Id')
test_data = pd.read_csv('/Users/houetsu/Downloads/Python/Kaggle/Houe_Price/Original_Data/test.csv', index_col='Id')

house_df.head()

print('train_data shape: ', house_df.shape)
print('test_data shape: ', test_data.shape)

# change certain numeric columns to categorical columns
num_to_cat = ['MoSold', 'YrSold', 'MSSubClass']
for col1 in num_to_cat:
    house_df[col1] = house_df[col1].astype('str', copy=False)

for col1 in num_to_cat:
    test_data[col1] = test_data[col1].astype('str', copy=False)



## Outliers ##
# check the correlation and decide which variable to use to define outliers
# correlation between numerical variables
corrmap = house_df.corr()
plt.figure(figsize=(30,15))
sns.heatmap(corrmap,annot=True,cmap="RdYlGn")
plt.show()

# choose GrLivArea
sns.scatterplot(x = 'GrLivArea', y = 'SalePrice', data = house_df)
plt.show()

house_df[(house_df['GrLivArea'] > 4000) & (house_df['SalePrice'] < 300000)]

house_df = house_df.drop(house_df[(house_df['GrLivArea'] > 4000) & (house_df['SalePrice'] < 300000)].index)



## organize response variable ##
sns.distplot(a = house_df['SalePrice'])
plt.show()

sns.boxplot(y = train_data['SalePrice'])
plt.show()


# log1p reverse function: expm1
sns.distplot(a = np.log1p(train_data['SalePrice']))
plt.show()

y = np.log1p(train_data['SalePrice'])



## check missing value and null ##
# train_data: calculate how many missng value of each columns and sort them from the maximum one
train_total_null = train_data.isnull().sum().sort_values(ascending = False)
# calculate the percentage
# in Python, Ture = 1 and False = 0, that's why we use sum and count to calculate the percentage
train_null_percent = (train_data.isnull().sum()/train_data.isnull().count()).sort_values(ascending = False)
# show the missing percentage and num
train_missing_value = pd.concat([train_total_null, train_null_percent], axis = 1, keys = ['train_total_null', 'train_null_percent'])
train_missing_value.head(20)

# test_data missing value
test_total_null = test_data.isnull().sum().sort_values(ascending = False)
test_null_percent = (test_data.isnull().sum()/test_data.isnull().count()).sort_values(ascending = False)
test_missing_value = pd.concat([test_total_null, test_null_percent], axis = 1, keys = ['test_total_null', 'test_null_percent'])
test_missing_value.head(36)

# null related variables
sp = [i for i in train_data['PoolArea'] if i == 0]
print('number of PoolArea == 0: ', len(sp))
sm = [i for i in train_data['MiscVal'] if i == 0]
print('number of MiscVal == 0: ', len(sm))
sf = [i for i in train_data['Fireplaces'] if i == 0]
print('number of Fireplaces == 0: ', len(sf))
sg = [i for i in train_data['GarageArea'] if i == 0]
print('number of GarageArea == 0: ', len(sg))


# I'll fill FireplaceQu null with 0
drop_missing = ['PoolQC', 'PoolArea', 'MiscFeature', 'MiscVal', 'Alley', 'Fence']

# inapproperiate columns which could lead to data leakage
# any variable updated (or created) after the target value is realized should be excluded
unavailable_variables = ['SaleType', 'SaleCondition']

# total drop variables
drop_var = drop_missing + unavailable_variables



## deal with missing value 
# FireplaceQu
house_df['FireplaceQu'] = house_df['FireplaceQu'].fillna('None')

# garage related
garage_cat = ['GarageType', 'GarageCond', 'GarageFinish', 'GarageQual']
house_df[garage_cat] = house_df[garage_cat].fillna('None')

garage_num = ['GarageYrBlt', 'GarageArea', 'GarageCars']
house_df[garage_num] = house_df[garage_num].fillna(0)

# basement related
bsmt_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
house_df[bsmt_cat] = house_df[bsmt_cat].fillna('None')

bsmt_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
house_df[bsmt_num] = house_df[bsmt_num].fillna(0)

# Masonry related
house_df['MasVnrType'] = house_df['MasVnrType'].fillna('None')

house_df['MasVnrArea'] = house_df['MasVnrArea'].fillna(0)

# mode fill
mode_fill = ['Electrical', 'Utilities', 'Functional', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'MSZoning']
for col in mode_fill:
	house_df[col] = house_df[col].fillna(house_df[col].mode()[0])


# fill test data
test_data['FireplaceQu'] = test_data['FireplaceQu'].fillna('None')

# garage related
garage_cat = ['GarageType', 'GarageCond', 'GarageFinish', 'GarageQual']
test_data[garage_cat] = test_data[garage_cat].fillna('None')

garage_num = ['GarageYrBlt', 'GarageArea', 'GarageCars']
test_data[garage_num] = test_data[garage_num].fillna(0)

# basement related
bsmt_cat = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
test_data[bsmt_cat] = test_data[bsmt_cat].fillna('None')

bsmt_num = ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']
test_data[bsmt_num] = test_data[bsmt_num].fillna(0)

# Masonry related
test_data['MasVnrType'] = test_data['MasVnrType'].fillna('None')

test_data['MasVnrArea'] = test_data['MasVnrArea'].fillna(0)

# mode fill
mode_fill = ['Electrical', 'Utilities', 'Functional', 'Exterior2nd', 'Exterior1st', 'KitchenQual', 'MSZoning']
for col in mode_fill:
	test_data[col] = test_data[col].fillna(test_data[col].mode()[0])


# LotFrontage
plt.subplots(figsize=(20, 6))
sns.swarmplot(x = 'Neighborhood', y = 'LotFrontage', data = train_data)
plt.show()

house_df['LotFrontage'] = house_df.groupby('Neighborhood')['LotFrontage'].transform(lambda x: 
	x.fillna(x.median()))

test_data['LotFrontage'] = test_data.groupby('Neighborhood')['LotFrontage'].transform(lambda x: 
	x.fillna(x.median()))



## Pivot table ##
house_df = house_df.drop(drop_var, axis = 1)

test_data = test_data.drop(drop_var, axis = 1)

house_cat = house_df.select_dtypes(include = 'object')

mlt = pd.melt(house_df, id_vars = house_cat.columns, value_vars = ['SalePrice'])
mlt.head()

for var in house_cat:
	a = (house_df.groupby(var)[var].count()/house_df.SalePrice.count()).round(4)*100
	b = pd.pivot_table(mlt, values = 'value', index = var, aggfunc = np.mean, margins = True)
	d = pd.pivot_table(mlt, values = 'value', index = var, aggfunc = np.std, margins = True)
	c = pd.concat([a,b,d], axis = 1, join = 'outer')
	display(c)


bad_cat = ['Street', 'Utilities', 'Condition2', 'RoofMatl', 'Heating']

house_df = house_df.drop(bad_cat, axis = 1)

test_data = test_data.drop(bad_cat, axis = 1)



## Numeric features ##
# correlation between numerical variables
corrmap = house_df.corr()
plt.figure(figsize=(30,15))
sns.heatmap(corrmap,annot=True,cmap="RdYlGn")
plt.show()


## check 0 of numeric variables
zero_count = (1460 - house_df.astype(bool).sum(axis=0)).sort_values(ascending = False)
zero_percent = ((1460-house_df.astype(bool).sum(axis=0))/house_df.astype(bool).count(axis=0)).sort_values(ascending = False)
house_zero = pd.concat([zero_count, zero_percent], axis = 1, keys = ['zero_count', 'zero_percent'])
house_zero.head(25)


## group related features
# area
# total finished basement square feet
house_df['BsmtFinSF'] = house_df['BsmtFinSF1'] + house_df['BsmtFinSF2'] 

# total house squrae feet
house_df['TotalHouseSF'] = house_df['TotalBsmtSF'] + house_df['GrLivArea'] 

# whether has more than one floor
def is1fl(row):
    if row['2ndFlrSF'] > 0:
        return 1
    return 0
house_df['is_over_1fl'] = house_df.apply(lambda row: is1fl(row), axis=1)


# porch area
house_df['TotalPorchSF'] = house_df['OpenPorchSF'] + house_df['EnclosedPorch'] + house_df['3SsnPorch'] 
+ house_df['ScreenPorch']

# bathrooms
house_df['TotalBath'] = house_df['BsmtFullBath'] + 0.5*house_df['BsmtHalfBath'] 
+ house_df['FullBath'] + 0.5*house_df['HalfBath']

# year remode and build
house_df['YearBuiltRemodeAvg'] = (house_df['YearBuilt'] + house_df['YearRemodAdd'])/2


# test_data
# area
# total finished basement square feet
test_data['BsmtFinSF'] = test_data['BsmtFinSF1'] + test_data['BsmtFinSF2'] 

# total house squrae feet
test_data['TotalHouseSF'] = test_data['TotalBsmtSF'] + test_data['GrLivArea'] 

# whether has more than one floor
def is1fl(row):
    if row['2ndFlrSF'] > 0:
        return 1
    return 0
test_data['is_over_1fl'] = test_data.apply(lambda row: is1fl(row), axis=1)


# porch area
test_data['TotalPorchSF'] = test_data['OpenPorchSF'] + test_data['EnclosedPorch'] + test_data['3SsnPorch'] 
+ test_data['ScreenPorch']

# bathrooms
test_data['TotalBath'] = test_data['BsmtFullBath'] + 0.5*test_data['BsmtHalfBath'] 
+ test_data['FullBath'] + 0.5*test_data['HalfBath']

# year remode and build
test_data['YearBuiltRemodeAvg'] = (test_data['YearBuilt'] + test_data['YearRemodAdd'])/2

test_data.shape


group_zero_drop_var = ['BsmtFinSF1', 'BsmtFinSF2', '2ndFlrSF', '1stFlrSF', 'OpenPorchSF', 'EnclosedPorch',
'3SsnPorch', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath']
house_df1 = house_df.drop(group_zero_drop_var, axis = 1)

test_data = test_data.drop(group_zero_drop_var, axis = 1)

house_df.columns

zero_count = (1460 - house_df.astype(bool).sum(axis=0)).sort_values(ascending = False)
zero_percent = ((1460-house_df.astype(bool).sum(axis=0))/house_df.astype(bool).count(axis=0)).sort_values(ascending = False)
house_zero = pd.concat([zero_count, zero_percent], axis = 1, keys = ['zero_count', 'zero_percent'])
house_zero.head(15)


# correlation between numerical variables
corrmap = house_df.corr()
plt.figure(figsize=(30,15))
sns.heatmap(corrmap,annot=True,cmap="RdYlGn")
plt.show()



## transform features ##
# check the distribution of each numeric feature
house_num = house_df.select_dtypes(exclude = 'object').drop('SalePrice', axis = 1)
for column in house_num:
    sns.distplot(a = house_df[column])
    plt.show()

## skewness
house_num = house_df.select_dtypes(exclude = 'object').drop('SalePrice', axis = 1)

skewness = house_num.apply(lambda x: skew(x)).sort_values(ascending=False)

skewness_df = pd.DataFrame({'Skewness': skewness})
skewness_df


(abs(skewness) > 0.75).value_counts()


boxcox_features = skewness_df[np.abs(skewness_df['Skewness'])>0.75].index
lam = 0.15

for col in boxcox_features:
    house_df.loc[:, col] = boxcox1p(house_df[col], lam)

for col in boxcox_features:
    test_data.loc[:, col] = boxcox1p(test_data[col], lam)


house_num_col = house_num.columns
skewness2 = house_df[house_num_col].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)

skewness_df2 = pd.DataFrame({'Skewness': skewness2})
skewness_df2.head(10)

house_df.head()

# correlation between numerical variables
corrmap = house_df.corr()
plt.figure(figsize=(30,15))
sns.heatmap(corrmap,annot=True,cmap="RdYlGn")
plt.show()



## Get dummmies ##
train_test_combine = pd.concat([house_df.drop('SalePrice', axis = 1),test_data], axis = 0, join = 'outer')

train_test_combine.shape

train_test_combine.tail() 

X_train_test = pd.get_dummies(train_test_combine)

X_train_test.shape

house_df.shape

X_final = X_train_test.iloc[:1458,:]

X_final.shape

X_test = X_train_test.iloc[1458:,:]

X_test.shape

X_test.head()



## Model ##
X_train, X_valid, y_train, y_valid = train_test_split(X_final, y, train_size=0.8, random_state=1)

# Define the model
my_model = XGBRegressor(n_estimators = 1000, learning_rate = 0.05) 

# Fit the model
my_model.fit(X_train, y_train, early_stopping_rounds = 5, eval_set = [(X_valid, y_valid)]) 

# Get predictions
y_predict = my_model.predict(X_valid) # Your code here


regressionSummary(y_predict, y_valid)


SalePriceArray = my_model.predict(X_test) # Your code here

SalePrice_final = np.expm1(SalePriceArray)


output = pd.DataFrame({'Id': test_data.index,
                      'SalePrice': SalePrice_final})
output.to_csv('submission.csv', index=False)

output.head()











