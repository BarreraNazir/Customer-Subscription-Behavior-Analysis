#!/usr/bin/env python
# coding: utf-8

# In[88]:


# Importing Libraries
import numpy as np # used for numeric calculations
import pandas as pd # used for data analysis and manipulation
import matplotlib.pyplot as plt # used for data visualization
import seaborn as sns 
from dateutil  import parser # used for converting time into date time datatype
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[25]:


# importing dataset
Finance_Data = pd.read_csv("B:\FineTech_appData.csv")


# In[26]:


features = Finance_Data.columns
for i in features:
    print("""Unique value = {}\n{}\nTotal Length = {} \n........................\n
          """.format(i, Finance_Data[i].unique(), len(Finance_Data[i].unique())))


# In[27]:


#reading first 5 rows of dataset
Finance_Data.head(5)


# In[29]:


# reading the data types of features
Finance_Data.info()


# In[36]:


# analyzing the data for null values
Finance_Data.isnull().sum()


# In[38]:


Finance_Data.dtypes


# In[40]:


# for data visualizing we need numeric data type.
# creating another variable to store numeric values
Finance_Data2 = Finance_Data.drop(['user', 'first_open', 'screen_list', 'enrolled_date'], axis = 1)


# In[41]:


# reading another numeric varibale
Finance_Data2.head(5)


# In[42]:


# Data Visualization by HeatMap
plt.figure(figsize=(16,9)) # heatmap size - ratio 16:9
 
sns.heatmap(Finance_Data2.corr(), annot = True, cmap ='coolwarm') # show heatmap
 
plt.title("Heatmap using correlation matrix of Finance_Data2", fontsize = 25) # title of heatmap


# In[43]:


# Data Visualization by Pairplot
sns.pairplot(Finance_Data2, hue = 'enrolled')


# In[49]:


print("User Enrolled = ", 50000 - (Finance_Data2.enrolled < 1).sum())
sns.countplot(Finance_Data2.enrolled)
print("User Not Enrolled = ",(Finance_Data2.enrolled < 1).sum())


# In[52]:


# Data Visualization via Histogram
plt.figure(figsize=(16,9)) # figure size - ratio 16:9
features = Finance_Data2.columns
for i,j in enumerate(features):
    plt.subplot(3,3,i+1) # creating subplot for histogram
    plt.title("Histogram of {}".format(j), fontsize = 15) # title of histogram
    
    bins = len(Finance_Data2[j].unique()) # bins of histogram
    plt.hist(Finance_Data2[j],bins = bins, rwidth = 0.8, edgecolor = "y", linewidth = 2, ) # plot histogram

plt.subplots_adjust(hspace=0.5)


# In[53]:


# Data Visualization via Bar Plot
sns.set # set background dark grid
plt.figure(figsize = (14,5))
plt.title("Correlating numeric features with 'enrolled' ", fontsize = 20)
Finance_Data3 = Finance_Data2.drop(['enrolled'], axis = 1)
ax = sns.barplot(Finance_Data3.columns, Finance_Data3. corrwith(Finance_Data.enrolled))
ax.tick_params(labelsize = 15, labelrotation = 20, color = "k")


# In[54]:


Finance_Data['first_open'] =[parser.parse(i) for i in Finance_Data['first_open']]
 
Finance_Data['enrolled_date'] =[parser.parse(i) if isinstance(i, str) else i for i in Finance_Data['enrolled_date']]
 
Finance_Data.dtypes


# In[55]:


Finance_Data['time_to_enrolled']  = (Finance_Data.enrolled_date - Finance_Data.first_open).astype('timedelta64[h]')
plt.hist(Finance_Data['time_to_enrolled'].dropna())


# In[56]:


plt.hist(Finance_Data['time_to_enrolled'].dropna(), range = (0,100)) 


# In[57]:


# Feature Selection
Finance_Data3 = pd.read_csv("top_screens.csv").top_screens.values
 
Finance_Data3


# In[58]:


Finance_Data['screen_list'] = Finance_Data.screen_list.astype(str) + ','


# In[59]:


# string into to number
 
for screen_name in Finance_Data3:
    Finance_Data[screen_name] = Finance_Data.screen_list.str.contains(screen_name).astype(int)
    Finance_Data['screen_list'] = Finance_Data.screen_list.str.replace(screen_name+",", "")


# In[60]:


Finance_Data.shape


# In[61]:


# remain screen in 'screen_list'
Finance_Data.loc[0,'screen_list']


# In[62]:


# count remain screen list and store counted number in 'remain_screen_list'
 
Finance_Data['remain_screen_list'] = Finance_Data.screen_list.str.count(",")


# In[63]:


# Drop the 'screen_list'
Finance_Data.drop(columns = ['screen_list'], inplace=True)


# In[64]:


Finance_Data.columns


# In[65]:


# taking sum of all saving screen in one place
saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',
                 ]
Finance_Data['saving_screens_count'] = Finance_Data[saving_screens].sum(axis = 1)
Finance_Data.drop(columns = saving_screens, inplace = True)


# In[66]:


# taking sum of all credit screen in one place
credit_screens = ['Credit1',
                  'Credit2',
                  'Credit3',
                  'Credit3Container',
                  'Credit3Dashboard',
                 ]
Finance_Data['credit_screens_count'] = Finance_Data[credit_screens].sum(axis = 1)
Finance_Data.drop(columns = credit_screens, axis = 1, inplace = True)


# In[67]:


# taking sum of all cc screen in one place
cc_screens = ['CC1',
              'CC1Category',
              'CC3',
             ]
Finance_Data['cc_screens_count'] = Finance_Data[cc_screens].sum(axis = 1)
Finance_Data.drop(columns = cc_screens, inplace = True)


# In[68]:


# taking sum of all Loan screen in one place
loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4',
               ]
Finance_Data['loan_screens_count'] = Finance_Data[loan_screens].sum(axis = 1)
Finance_Data.drop(columns = loan_screens, inplace = True)


# In[69]:


Finance_Data.shape


# In[70]:


Finance_Data.info()


# In[71]:


# Drop the 'screen_list'
Finance_Data.drop(columns = ['first_open'], inplace=True)


# In[73]:


Finance_Data.drop(columns = ['enrolled_date'], inplace=True)


# In[74]:


Finance_Data.drop(columns = ['time_to_enrolled'], inplace=True)


# In[75]:


Finance_Data.info()


# In[107]:


traget = Finance_Data['enrolled']
Finance_Data.drop(columns = 'enrolled', inplace = True)


# In[82]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Finance_Data, traget, test_size = 0.2, random_state = 0)

print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of y_test = ', y_test.shape)


# In[85]:


# take User ID in another variable 
train_userID = X_train['user']
X_train.drop(columns= 'user', inplace =True)
test_userID = X_test['user']
X_test.drop(columns= 'user', inplace =True)


# In[86]:


print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of train_userID = ', train_userID.shape)
print('Shape of test_userID = ', test_userID.shape)


# In[87]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[89]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)
accuracy_score(y_test, y_pred_dt)


# In[90]:


# K- Nearest Neighbor Classifier
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
 
accuracy_score(y_test, y_pred_knn)


# In[91]:


# Naive Bayes Classifier
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)
 
accuracy_score(y_test, y_pred_nb)


# In[92]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
 
accuracy_score(y_test, y_pred_rf)


# In[93]:


# Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state = 0, penalty = 'l1')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
 
accuracy_score(y_test, y_pred_lr)


# In[94]:


# Support Vector Machine
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)
 
accuracy_score(y_test, y_pred_svc)


# In[95]:


# XGBoost Classifier
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
accuracy_score(y_test, y_pred_xgb)


# In[99]:


# Among all classifiers, XGBoost ML Model gave better results 


# In[101]:


# confussion matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
sns.heatmap(cm_xgb, annot = True, fmt = 'g')
plt.title("Confussion Matrix", fontsize = 20) 


# In[102]:


# Clasification Report
cr_xgb = classification_report(y_test, y_pred_xgb)
 
print("Classification report >>> \n", cr_xgb)


# In[104]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_model, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())


# In[103]:


final_result = pd.concat([test_userID, y_test], axis = 1)
final_result['predicted result'] = y_pred_xgb
 
print(final_result)

