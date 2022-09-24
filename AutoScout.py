#!/usr/bin/env python
# coding: utf-8

# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___

# # WELCOME!

# Welcome to "***Car Price Prediction Project***". This is the first medium project of ***Machine Learning*** course. In this project you will have the opportunity to apply many algorithms commonly used for regression problems.
# 
# Also, you will apply various processes such as pre-processing, ***train-test spilit*** and ***cross validation*** that you will use in algorithm modeling and prediction processes in Python with ***scikit-learn***. 
# 
# Before diving into the project, please take a look at the determines and tasks.
# 
# - **NOTE:** This project assumes that you already know the basics of coding in Python. You should also be familiar with the theory behind regression algorithms and scikit-learn module as well as Machine Learning before you begin.
# 

# # #Determines
# 
# **Auto Scout** data which using for this project, scraped from the on-line car trading company(https://www.autoscout24.com)in 2019, contains many features of 9 different car models. In this project, you will use the data set which is already preprocessed and prepared for algorithms .
# 
# The aim of this project to understand of machine learning algorithms. Therefore, you will not need any EDA process as you will be working on the edited data.
# 
# ---
# 
# In this Senario, you will estimate the prices of cars using regression algorithms.
# 
# While starting you should import the necessary modules and load the data given as pkl file. Also you'll need to do a few pre-processing before moving to modelling. After that you will implement ***Linear Regression, Ridge Regression, Lasso Regression,and Elastic-Net algorithms respectively*** (After completion of Unsupervised Learning section, you can also add bagging and boosting algorithms such as ***Random Forest and XG Boost*** this notebook to develop the project. You can measure the success of your models with regression evaluation metrics as well as with cross validation method.
# 
# For the better results, you should try to increase the success of your models by performing hyperparameter tuning. Determine feature importances for the model. You can set your model with the most important features for resource saving. You should try to apply this especially in Random Forest and XG Boost algorithms. Unlike the others, you will perform hyperparameter tuning for Random Forest and XG Boost using the ***GridSearchCV*** method. 
# 
# Finally You can compare the performances of algorithms, work more on the algorithm have the most successful prediction rate.
# 
# 
# 
# 
# 

# # #Tasks
# 
# #### 1. Import Modules, Load Data and Data Review
# #### 2. Data Pre-Processing
# #### 3. Implement Linear Regression 
# #### 4. Implement Ridge Regression
# #### 5. Implement Lasso Regression 
# #### 6. Implement Elastic-Net
# #### 7. Visually Compare Models Performance In a Graph

# ## 1. Import Modules, Load Data and Data Review

# In[1]:


import pandas as pd      
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from scipy.stats import skew

from sklearn.model_selection import cross_validate
import warnings
warnings.filterwarnings('ignore')
plt.rcParams["figure.figsize"] = (10,6)
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)


# In[2]:


df = pd.read_csv("final_scout_not_dummy.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isnull().sum()


# In[7]:


df.make_model


# In[8]:


df2 = df.copy()


# In[ ]:





# ## Feature Engineering

# In[9]:


df_object = df.select_dtypes(include ="object").head()
df_object


# In[10]:


for col in df_object:
    print(f"{col:<20}:", df[col].nunique())


# In[11]:


df.make_model.value_counts()


# In[12]:


ax = df.make_model.value_counts().plot(kind ="bar")

ax.bar_label(ax.containers[0]);


# In[13]:


df[df.make_model=="Audi A2"]


# In[14]:


df.drop(index=[2614], inplace =True)


# In[15]:


df.shape


# In[16]:


sns.histplot(df.price, bins=50, kde=True)


# In[17]:


skew(df.price)


# In[18]:


df_numeric = df.select_dtypes(include ="number")
df_numeric


# In[19]:


sns.heatmap(df_numeric.corr(), annot =True)


# ## multicollinearity control

# In[20]:


df_numeric.corr()[(df_numeric.corr()>= 0.9) & (df_numeric.corr() < 1)].any().any()


# In[21]:


df_numeric.corr()[(df_numeric.corr()<= -0.9) & (df_numeric.corr() > -1)].any().any()


# In[22]:


sns.boxplot(df.price)


# In[23]:


plt.figure(figsize=(16,6))
sns.boxplot(x="make_model", y="price", data=df, whis=1.5)
plt.show()


# In[24]:


df[df["make_model"]== "Audi A1"]["price"]


# In[25]:


total_outliers = []

for model in df.make_model.unique():
    
    car_prices = df[df["make_model"]== model]["price"]
    
    Q1 = car_prices.quantile(0.25)
    Q3 = car_prices.quantile(0.75)
    IQR = Q3-Q1
    lower_lim = Q1-1.5*IQR
    upper_lim = Q3+1.5*IQR
    
    count_of_outliers = (car_prices[(car_prices < lower_lim) | (car_prices > upper_lim)]).count()
    
    total_outliers.append(count_of_outliers)
    
    print(f" The count of outlier for {model:<15} : {count_of_outliers:<5},           The rate of outliers : {(count_of_outliers/len(df[df['make_model']== model])).round(3)}")
print()    
print("Total_outliers : ",sum(total_outliers), "The rate of total outliers :", (sum(total_outliers)/len(df)).round(3))


# ## 2. Data Pre-Processing
# 
# As you know, the data set must be edited before proceeding to the implementation of the model. As the last step before model fitting, you need to spilit the data set as train and test. Then, you should train the model with train data and evaluate the performance of the model on the test data. You can use the train and test data you have created for all algorithms.
# 
# You must also drop your target variable, the column you are trying to predict.
# 
# You can use many [performance metrics for regression](https://medium.com/analytics-vidhya/evaluation-metrics-for-regression-problems-343c4923d922) to measure the performance of the regression model you train. You can define a function to view different metric results together.
# 
# You can also use the [cross validation](https://towardsdatascience.com/cross-validation-explained-evaluating-estimator-performance-e51e5430ff85) method to measure the estimator performance. Cross validation uses different data samples from your test set and calculates the accuracy score for each data sample. You can calculate the final performance of your estimator by averaging these scores.

# ### Train | Test Split

# In[26]:


X= df.drop(columns="price")
y= df.price


# In[27]:


def trans_1(X, y, test_size = 0.2, random_state=101):
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    X_train = X_train.join(X_train["Comfort_Convenience"].str.get_dummies(sep = ",").add_prefix("cc_"))
    X_train = X_train.join(X_train["Entertainment_Media"].str.get_dummies(sep = ",").add_prefix("em_"))
    X_train = X_train.join(X_train["Extras"].str.get_dummies(sep = ",").add_prefix("ex_"))
    X_train = X_train.join(X_train["Safety_Security"].str.get_dummies(sep = ",").add_prefix("ss_"))
    
    
    X_test = X_test.join(X_test["Comfort_Convenience"].str.get_dummies(sep = ",").add_prefix("cc_"))
    X_test = X_test.join(X_test["Entertainment_Media"].str.get_dummies(sep = ",").add_prefix("em_"))
    X_test = X_test.join(X_test["Extras"].str.get_dummies(sep = ",").add_prefix("ex_"))
    X_test = X_test.join(X_test["Safety_Security"].str.get_dummies(sep = ",").add_prefix("ss_"))
    
    X_test = X_test.reindex(columns = X_train.columns, fill_value=0) # "0"
    
    
    X_train.drop(columns=["Comfort_Convenience","Entertainment_Media","Extras","Safety_Security"], inplace = True)
    X_test.drop(columns=["Comfort_Convenience","Entertainment_Media","Extras","Safety_Security"], inplace = True)
    
    
    return X_train, X_test, y_train, y_test


# ### Example for reindex

# In[28]:


train = {"a": [1, 2], "b": [2,3], "c":[1,4], "d":[2,4], "e":[5,6]}
test = {"e": [1, 2], "c": [2,3], "a":[1,4], "d":[2,4]}
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train


# In[29]:


test


# In[30]:


test.reindex(columns = train.columns, fill_value=0)


# In[31]:


X_train, X_test, y_train, y_test = trans_1(X, y)


# In[32]:


X_train.head()


# In[33]:


X_test.head()


# ## OneHotEncoder

# ### Example

# In[34]:


from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder(handle_unknown="ignore", sparse=False)


# In[35]:


data =pd.DataFrame(pd.Series(['good','bad','worst','good', 'good', 'bad', 'bed']))
# new_data has two values that data does not have. 
new_data= pd.DataFrame(pd.Series(['bad','worst','good', 'good', 'bad', "bed", "resume", "car"]))


# In[36]:


data


# In[37]:


new_data


# In[38]:


enc.fit_transform(data[[0]])


# In[39]:


enc.transform(new_data[[0]])


# In[40]:


enc.get_feature_names_out(["0"])


# In[41]:


pd.DataFrame(enc.fit_transform(data[[0]]), columns = enc.get_feature_names_out(["0"]))


# In[42]:


pd.DataFrame(enc.transform(new_data[[0]]), columns = enc.get_feature_names_out(["0"]))


# In[ ]:





# ### OneHotEncoder for X_train and X_test

# In[43]:


cat = X_train.select_dtypes("object").columns
cat


# In[44]:


cat = list(cat)
cat


# In[45]:


enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names_out(cat))


# In[46]:


enc.fit_transform(X_train[cat])


# In[47]:


enc.get_feature_names_out(cat)


# In[48]:


X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names_out(cat))
X_train_cat


# In[49]:


X_train.select_dtypes("number")


# In[50]:


X_train_new = X_train_cat.join(X_train.select_dtypes("number"))
X_train_new


# In[51]:


X_train_new


# In[ ]:





# In[ ]:





# In[52]:


X_test_cat = pd.DataFrame(enc.transform(X_test[cat]), index = X_test.index, columns = enc.get_feature_names_out(cat))
X_test_cat


# In[53]:


X_test.select_dtypes("number")


# In[54]:


X_test_new = X_test_cat.join(X_test.select_dtypes("number"))
X_test_new


# In[55]:


X_test_new


# In[ ]:





# In[ ]:





# In[59]:


def trans_2(X_train, X_test):
    
    cat = X_train.select_dtypes("object").columns
    cat = list(cat)
    
    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    
    X_train_cat = pd.DataFrame(enc.fit_transform(X_train[cat]), index = X_train.index, 
                           columns = enc.get_feature_names_out(cat))
    
    X_test_cat  = pd.DataFrame(enc.transform(X_test[cat]), index = X_test.index, 
                               columns = enc.get_feature_names_out(cat))
    
    X_train = X_train_cat.join(X_train.select_dtypes("number"))
    X_test = X_test_cat.join(X_test.select_dtypes("number"))
    
    
    return X_train, X_test


# In[60]:


X_train, X_test = trans_2(X_train, X_test)


# In[61]:


X_train.head()


# In[62]:


X_test.head()


# In[63]:


corr_by_price = X_train.join(y_train).corr()["price"].sort_values()[:-1]
corr_by_price


# In[64]:


plt.figure(figsize = (20,10))
sns.barplot(x = corr_by_price.index, y = corr_by_price)
plt.xticks(rotation=90)
plt.tight_layout();


# ## 3. Implement Linear Regression

#  - Import the modul
#  - Fit the model 
#  - Predict the test set
#  - Determine feature coefficiant
#  - Evaluate model performance (use performance metrics for regression and cross_val_score)
#  - Compare different evaluation metrics
#  
# *Note: You can use the [dir()](https://www.geeksforgeeks.org/python-dir-function/) function to see the methods you need.*

# In[65]:


def train_val(model, X_train, y_train, X_test, y_test):
    
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    scores = {"train": {"R2" : r2_score(y_train, y_train_pred),
    "mae" : mean_absolute_error(y_train, y_train_pred),
    "mse" : mean_squared_error(y_train, y_train_pred),                          
    "rmse" : np.sqrt(mean_squared_error(y_train, y_train_pred))},
    
    "test": {"R2" : r2_score(y_test, y_pred),
    "mae" : mean_absolute_error(y_test, y_pred),
    "mse" : mean_squared_error(y_test, y_pred),
    "rmse" : np.sqrt(mean_squared_error(y_test, y_pred))}}
    
    return pd.DataFrame(scores)


# In[66]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,y_train)


# In[67]:


pd.options.display.float_format = '{:.3f}'.format


# In[69]:


train_val(lm, X_train, y_train, X_test, y_test)


# ## Adjusted R2 Score

# In[70]:


def adj_r2(y_test, y_pred, X):
    r2 = r2_score(y_test, y_pred)
    n = X.shape[0]   # number of observations
    p = X.shape[1]   # number of independent variables 
    adj_r2 = 1 - (1-r2)*(n-1)/(n-p-1)
    return adj_r2


# In[71]:


y_pred = lm.predict(X_test)


# In[72]:


adj_r2(y_test, y_pred, X)


# ## Cross Validate

# In[73]:


model = LinearRegression()
scores = cross_validate(model, X_train, y_train, scoring=['r2', 
            'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv =10)


# In[74]:


pd.DataFrame(scores)


# In[75]:


pd.DataFrame(scores).iloc[:, 2:].mean()


# In[76]:


train_val(lm, X_train, y_train, X_test, y_test)


# In[77]:


2405/df.price.mean()


# ## Prediction Error

# In[78]:


from yellowbrick.regressor import PredictionError
from yellowbrick.features import RadViz

visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();


# ## Residual Plot

# In[79]:


plt.figure(figsize=(12,8))
residuals = y_test-y_pred

sns.scatterplot(x = y_test, y = -residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()


# In[80]:


sns.kdeplot(residuals)


# In[81]:


skew(residuals)


# In[82]:


from yellowbrick.regressor import ResidualsPlot

visualizer = RadViz(size=(1000, 720))
model = LinearRegression()
visualizer = ResidualsPlot(model)

visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();       


# ## Dropping outliers that worsen my predictions from the dataset

# In[83]:


for model in df2.make_model.unique():
    
    car_prices = df2[df2["make_model"]== model]["price"]
    
    Q1 = car_prices.quantile(0.25)
    Q3 = car_prices.quantile(0.75)
    
    IQR = Q3-Q1
    
    lower_lim = Q1-1.5*IQR
    upper_lim = Q3+1.5*IQR

    drop_index = df2[df2["make_model"]== model][(car_prices < lower_lim) | (car_prices > upper_lim)].index
    df2.drop(index = drop_index, inplace=True)
df2


# In[84]:


15496+419


# In[85]:


df2[df2.make_model=="Audi A2"]


# In[86]:


df2.drop(index=[2614], inplace =True)


# In[87]:


df2.reset_index(drop=True, inplace=True)


# In[88]:


df2


# In[89]:


df3 = df2.copy()


# In[90]:


X = df2.drop(columns = "price")
y = df2.price

X_train, X_test, y_train, y_test = trans_1(X, y)
X_train, X_test = trans_2(X_train, X_test)


# In[91]:


X_train.head()


# In[92]:


X_test.head()


# In[93]:


lm2 = LinearRegression()
lm2.fit(X_train,y_train)


# In[94]:


visualizer = RadViz(size=(720, 3000))
model = LinearRegression()
visualizer = PredictionError(model)
visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
visualizer.score(X_test, y_test)  # Evaluate the model on the test data
visualizer.show();


# In[95]:


plt.figure(figsize=(12,8))
y_pred = lm2.predict(X_test)
residuals = y_test-y_pred

sns.scatterplot(x = y_test, y = -residuals) #-residuals
plt.axhline(y = 0, color ="r", linestyle = "--")
plt.ylabel("residuals")
plt.show()


# In[96]:


train_val(lm2, X_train, y_train, X_test, y_test)

     train	      test
R2	 0.890	      0.890
mae	 1705.452	  1705.217
mse	 6038122.231  5785150.711
rmse 2457.259	  2405.234
# In[97]:


2052/df2.price.mean()


# In[98]:


2405/df.price.mean()


# In[99]:


model = LinearRegression()
scores = cross_validate(model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], cv=10)


# In[100]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:]


# In[101]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# In[102]:


train_val(lm2, X_train, y_train, X_test, y_test)


# In[103]:


y_pred = lm2.predict(X_test)

lm_R2 = r2_score(y_test, y_pred)
lm_mae = mean_absolute_error(y_test, y_pred)
lm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# In[104]:


my_dict = { 'Actual': y_test, 'Pred': y_pred, 'Residual': y_test-y_pred }
compare = pd.DataFrame(my_dict)


# In[105]:


comp_sample = compare.sample(20)
comp_sample


# In[106]:


comp_sample.plot(kind='bar',figsize=(15,9))
plt.show()


# In[107]:


pd.DataFrame(lm2.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")


# ## 4. Implement Ridge Regression

# - Import the modul 
# - Do not forget to scale the data or use Normalize parameter as True 
# - Fit the model 
# - Predict the test set 
# - Evaluate model performance (use performance metrics for regression) 
# - Tune alpha hiperparameter by using [cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.RidgeCV.html) and determine the optimal alpha value.
# - Fit the model and predict again with the new alpha value. 

# ## Scaling

# In[108]:


scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ## Ridge

# In[109]:


from sklearn.linear_model import Ridge


# In[110]:


ridge_model = Ridge()


# In[111]:


ridge_model.fit(X_train_scaled, y_train)


# In[112]:


train_val(ridge_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ## Cross Validation

# In[113]:


model = Ridge()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], 
                        cv=10)


# In[114]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# ## Finding best alpha for Ridge

# In[115]:


from sklearn.model_selection import GridSearchCV


# In[116]:


alpha_space = np.linspace(0.01, 100, 100)
alpha_space


# In[117]:


ridge_model = Ridge()

param_grid = {'alpha':alpha_space}

ridge_grid_model = GridSearchCV(estimator=ridge_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)


# In[118]:


ridge_grid_model.fit(X_train_scaled,y_train)


# In[119]:


#ridge_grid_model.best_estimator_


# In[120]:


ridge_grid_model.best_params_


# In[121]:


pd.DataFrame(ridge_grid_model.cv_results_)


# In[122]:


ridge_grid_model.best_index_


# In[123]:


ridge_grid_model.best_score_


# In[124]:


train_val(ridge_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)


# In[125]:


y_pred = ridge_grid_model.predict(X_test_scaled)
rm_R2 = r2_score(y_test, y_pred)
rm_mae = mean_absolute_error(y_test, y_pred)
rm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# In[126]:


ridge = Ridge(alpha=1.02).fit(X_train_scaled, y_train)

pd.DataFrame(ridge.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")


# ## 5. Implement Lasso Regression

# - Import the modul 
# - Do not forget to scale the data or use Normalize parameter as True(If needed)
# - Fit the model 
# - Predict the test set 
# - Evaluate model performance (use performance metrics for regression) 
# - Tune alpha hyperparameter by using [cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html) and determine the optimal alpha value.
# - Fit the model and predict again with the new alpha value.
# - Compare different evaluation metrics
# 
# *Note: To understand the importance of the alpha hyperparameter, you can observe the effects of different alpha values on feature coefficants.*

# In[127]:


from sklearn.linear_model import Lasso


# In[128]:


lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)


# In[129]:


train_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ## Cross Validation

# In[130]:


model = Lasso()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)


# In[131]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# ## Finding best alpha for Lasso

# In[132]:


lasso_model = Lasso()

param_grid = {'alpha':alpha_space}

lasso_grid_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)


# In[133]:


lasso_grid_model.fit(X_train_scaled,y_train)


# In[134]:


lasso_grid_model.best_params_


# In[135]:


lasso_grid_model.best_score_


# In[136]:


train_val(lasso_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)


# In[137]:


y_pred = lasso_grid_model.predict(X_test_scaled)
lasm_R2 = r2_score(y_test, y_pred)
lasm_mae = mean_absolute_error(y_test, y_pred)
lasm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# In[138]:


lasso = Lasso(alpha=1.02).fit(X_train_scaled, y_train)
pd.DataFrame(lasso.coef_, index = X_train.columns, columns=["Coef"]).sort_values("Coef")


# ## 6. Implement Elastic-Net

# - Import the modul 
# - Do not forget to scale the data or use Normalize parameter as True(If needed)
# - Fit the model 
# - Predict the test set 
# - Evaluate model performance (use performance metrics for regression) 
# - Tune alpha hyperparameter by using [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) and determine the optimal alpha value.
# - Fit the model and predict again with the new alpha value.
# - Compare different evaluation metrics

# In[139]:


from sklearn.linear_model import ElasticNet


# In[140]:


elastic_model = ElasticNet()
elastic_model.fit(X_train_scaled,y_train)


# In[141]:


train_val(elastic_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ## Cross Validation

# In[142]:


model = ElasticNet()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'], 
                        cv=10)


# In[143]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# ## Finding best alpha and l1_ratio for ElasticNet

# In[144]:


elastic_model = ElasticNet()


# In[145]:


param_grid = {'alpha':[1.02, 2,  3, 4, 5, 7, 10, 11],
              'l1_ratio':[.5, .7, .9, .95, .99, 1]}

elastic_grid_model = GridSearchCV(estimator=elastic_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)


# In[146]:


elastic_grid_model.fit(X_train_scaled,y_train)


# In[147]:


elastic_grid_model.best_params_


# In[148]:


elastic_grid_model.best_score_


# In[149]:


train_val(elastic_grid_model, X_train_scaled, y_train, X_test_scaled, y_test)


# In[150]:


y_pred = elastic_grid_model.predict(X_test_scaled)
em_R2 = r2_score(y_test, y_pred)
em_mae = mean_absolute_error(y_test, y_pred)
em_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# ## Feature İmportance

# In[151]:


from yellowbrick.model_selection import FeatureImportances
from yellowbrick.features import RadViz

model = Lasso(alpha=1.02)

viz = FeatureImportances(model, labels=X_train.columns)
visualizer = RadViz(size=(720, 3000))
viz.fit(X_train_scaled, y_train)
viz.show();


# In[ ]:





# In[152]:


df_new = df3[["make_model", "hp_kW", "km","age", "price", "Gearing_Type", "Gears"]]


# In[153]:


df_new


# In[154]:


X = df_new.drop(columns = ["price"])
y = df_new.price


# In[155]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[156]:


X_train, X_test = trans_2(X_train, X_test)


# In[157]:


X_train


# In[158]:


X_test


# In[159]:


scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[160]:


lasso_model = Lasso()
lasso_model.fit(X_train_scaled, y_train)
train_val(lasso_model, X_train_scaled, y_train, X_test_scaled, y_test)


# ## Cross Validate

# In[161]:


model = Lasso()
scores = cross_validate(model, X_train_scaled, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)


# In[162]:


scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# ## Gridsearch

# In[163]:


lasso_model = Lasso()

param_grid = {'alpha':alpha_space}

lasso_final_model = GridSearchCV(estimator=lasso_model,
                          param_grid=param_grid,
                          scoring='neg_root_mean_squared_error',
                          cv=10,
                          n_jobs = -1)


# In[ ]:


lasso_final_model.fit(X_train_scaled,y_train)


# In[ ]:


lasso_final_model.best_params_


# In[ ]:


lasso_final_model.best_score_


# In[ ]:


train_val(lasso_final_model, X_train_scaled, y_train, X_test_scaled, y_test)


# In[ ]:


2364/df_new.price.mean()


# In[ ]:


y_pred = lasso_final_model.predict(X_test_scaled)
fm_R2 = r2_score(y_test, y_pred)
fm_mae = mean_absolute_error(y_test, y_pred)
fm_rmse = np.sqrt(mean_squared_error(y_test, y_pred))


# ## 7. Visually Compare Models Performance In a Graph

# In[ ]:


scores = {"linear_m": {"r2_score": lm_R2 , 
 "mae": lm_mae, 
 "rmse": lm_rmse},

 "ridge_m": {"r2_score": rm_R2, 
 "mae": rm_mae,
 "rmse": rm_rmse},
    
 "lasso_m": {"r2_score": lasm_R2, 
 "mae": lasm_mae, 
 "rmse": lasm_rmse},

 "elastic_m": {"r2_score": em_R2, 
 "mae": em_mae, 
 "rmse": em_rmse},
         
 "final_m": {"r2_score": fm_R2, 
 "mae": fm_mae , 
 "rmse": fm_rmse}}
scores = pd.DataFrame(scores).T
scores


# In[ ]:


for i, j in enumerate(scores):
    print(i, j)


# In[ ]:


#metrics = scores.columns
for i, j in enumerate(scores):
    plt.figure(i)
    if j == "r2_score":
        ascending = False
    else:
        ascending = True
    compare = scores.sort_values(by=j, ascending=ascending)
    ax = sns.barplot(x = compare[j] , y= compare.index)
    ax.bar_label(ax.containers[0], fmt="%.4f");


# ## Prediction new observation

# In[ ]:


X = df_new.drop(columns = ["price"])
y = df_new.price


# In[ ]:


X.head()


# In[ ]:


cat = X.select_dtypes("object").columns
cat = list(cat)
cat


# In[ ]:


enc = OneHotEncoder(handle_unknown="ignore", sparse=False)

X_cat = pd.DataFrame(enc.fit_transform(X[cat]), index = X.index, 
                           columns = enc.get_feature_names_out(cat))

X = X_cat.join(X)
X.drop(columns = cat, inplace=True)
X


# In[ ]:


final_scaler = MinMaxScaler()
final_scaler.fit(X)
X_scaled = final_scaler.transform(X)


# In[ ]:


final_model = Lasso(alpha=0.01)


# In[ ]:


final_model.fit(X_scaled, y)


# In[ ]:


my_dict = {
    "hp_kW": 66,
    "age": 2,
    "km": 17000,
    "Gears": 7,
    "make_model": 'Audi A3',
    "Gearing_Type": "Automatic"
}


# In[ ]:


new_obs = pd.DataFrame([my_dict])
new_obs


# In[ ]:


onehot = pd.DataFrame(enc.transform(new_obs[cat]), index=new_obs.index,
                           columns = enc.get_feature_names_out(cat))
new_obs = onehot.join(new_obs)
new_obs.drop(columns = cat, inplace=True)
new_obs


# In[ ]:


new_obs = new_obs.reindex(columns=X.columns)
new_obs


# In[ ]:


new_obs = final_scaler.transform(new_obs)
new_obs


# In[ ]:


final_model.predict(new_obs)


# ## Pipeline

# In[ ]:


X = df_new.drop(columns = ["price"])
y = df_new.price


# In[ ]:


cat = X.select_dtypes("object").columns
cat = list(cat)
cat


# In[ ]:


X.head(1)


# In[ ]:


from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), 
                                       remainder=MinMaxScaler())


# In[ ]:


from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X, y)


# In[ ]:


my_dict = {
    "hp_kW": 66,
    "age": 2,
    "km": 17000,
    "Gears": 7,
    "make_model": 'Audi A3',
    "Gearing_Type": "Automatic"
    
}


# In[ ]:


new_obs = pd.DataFrame([my_dict])
new_obs


# In[ ]:


pipe_model.predict(new_obs)


# In[ ]:





# ## Cross Validate With Pipeline

# In[ ]:


X = df_new.drop(columns = ["price"])
y = df_new.price


# In[ ]:


X.head()


# In[ ]:


cat = X.select_dtypes("object").columns
cat = list(cat)
cat


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)


# In[ ]:


column_trans = make_column_transformer((OneHotEncoder(handle_unknown="ignore", sparse=False), cat), 
                                       remainder=MinMaxScaler())

operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]

pipe_model = Pipeline(steps=operations)

pipe_model.fit(X_train, y_train)


# In[ ]:


pipe_model.predict(new_obs)


# In[ ]:


train_val(pipe_model, X_train, y_train, X_test, y_test)


# In[ ]:


operations = [("OneHotEncoder", column_trans), ("Lasso", Lasso())]

pipe_model = Pipeline(steps=operations)

scores = cross_validate(pipe_model, X_train, y_train,
                        scoring=['r2', 'neg_mean_absolute_error','neg_mean_squared_error','neg_root_mean_squared_error'],
                        cv=10)
scores = pd.DataFrame(scores, index = range(1, 11))
scores.iloc[:,2:].mean()


# In[ ]:





# In[ ]:





# ___
# 
# <p style="text-align: center;"><img src="https://docs.google.com/uc?id=1lY0Uj5R04yMY3-ZppPWxqCr5pvBLYPnV" class="img-fluid" alt="CLRSWY"></p>
# 
# ___
