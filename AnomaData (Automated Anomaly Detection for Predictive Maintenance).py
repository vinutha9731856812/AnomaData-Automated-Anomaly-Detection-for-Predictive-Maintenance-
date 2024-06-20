#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV


# In[2]:


# Step 1: Loading the dataset
data = pd.read_excel("AnomaData.xlsx")


# In[3]:


# Step 2: Exploratory Data Analysis (EDA)
# Check data quality
print(data.info())
print(data.describe())


# In[4]:


# Treat missing values if any
data.dropna(inplace=True)


# In[5]:


# Step 3: Correcting date datatype
data['time'] = pd.to_datetime(data['time'])


# In[6]:


# Step 4: Feature Engineering and Selection
# Assuming no specific feature engineering required


# In[7]:


# Step 5: Train/Test Split
X = data.drop(columns=['y', 'y.1'])
y = data['y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[8]:


# Convert datetime to numerical representation
# Convert 'time' column to datetime if it's not already
X_train['time'] = pd.to_datetime(X_train['time'])
X_test['time'] = pd.to_datetime(X_test['time'])

# Convert datetime to Unix timestamp in seconds
X_train['time'] = X_train['time'].astype('int64') // 10**9
X_test['time'] = X_test['time'].astype('int64') // 10**9


# In[9]:


# Step 6: Model Selection
model = RandomForestClassifier()


# In[10]:


# Step 7: Model Training
model.fit(X_train, y_train)


# In[19]:


# Step 8: Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))


# In[12]:


# Step 9: Hyperparameter Tuning/Model Improvement
param_grid = {'n_estimators': [100, 200, 300],
              'max_depth': [None, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}


# In[13]:


grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)



# In[15]:


grid_search.fit(X_train, y_train)


# In[16]:


best_params = grid_search.best_params_
print("Best parameters:", best_params)


# In[ ]:




