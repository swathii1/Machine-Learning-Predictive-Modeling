#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os               
import numpy                   as np
import pandas                  as pd 
import matplotlib.pyplot       as plt
import seaborn                 as sns
import plotly.express          as ex
import plotly.graph_objs       as go
import plotly.offline          as pyo
import scipy.stats             as stats
import pymc3                   as pm
import theano.tensor           as tt
from plotly.subplots           import make_subplots
from sklearn.preprocessing     import StandardScaler
from sklearn.decomposition     import TruncatedSVD,PCA
from sklearn.ensemble          import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree              import DecisionTreeClassifier
sns.set_style('darkgrid')
pyo.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model      import LinearRegression,LogisticRegressionCV
from sklearn.svm               import SVC
from sklearn.metrics           import mean_squared_error,r2_score
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import cross_val_score,train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.manifold          import Isomap,TSNE
from sklearn.feature_selection import mutual_info_classif
from tqdm.notebook             import tqdm
from scipy.stats               import ttest_ind
plt.rc('figure',figsize=(18,11))
sns.set_context('paper',font_scale=2)
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from tensorflow.keras.callbacks import ModelCheckpoint
import re
import random


# In[ ]:


import os               
import numpy                   as np
import pandas                  as pd 
import matplotlib.pyplot       as plt
import seaborn                 as sns
import plotly.express          as ex
import plotly.graph_objs       as go
import plotly.offline          as pyo
import scipy.stats             as stats
#import pymc3                   as pm
#import theano.tensor           as tt
from plotly.subplots           import make_subplots
from sklearn.preprocessing     import StandardScaler
from sklearn.decomposition     import TruncatedSVD,PCA
from sklearn.ensemble          import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.tree              import DecisionTreeClassifier
from sklearn.linear_model      import LinearRegression,LogisticRegressionCV
from sklearn.svm               import SVC
from sklearn.metrics           import mean_squared_error,r2_score
from sklearn.pipeline          import Pipeline
from sklearn.model_selection   import cross_val_score,train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.manifold          import Isomap,TSNE
from sklearn.feature_selection import mutual_info_classif
from tqdm.notebook             import tqdm
from scipy.stats               import ttest_ind


# In[2]:


sns.set_style('darkgrid')
pyo.init_notebook_mode()
get_ipython().run_line_magic('matplotlib', 'inline')


plt.rc('figure',figsize=(18,11))
sns.set_context('paper',font_scale=2)


# In[4]:


#Read data

df=pd.read_csv('water_potability.csv')


# In[5]:


df


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.Potability.value_counts().plot(kind ='pie');


# In[9]:


df['Potability'].value_counts()


# In[10]:


#Null values

zero  = df[df['Potability']==0]   #zero values in Potability column
one = df[df['Potability']==1]  # one values in Potability column
from sklearn.utils import resample
#minority class that  is 1, we need to upsample/increase that class so that there is no bias
#n_samples = 1998 means we want 1998 sample of class 1, since there are 1998 samples of class 0
df_minority_upsampled = resample(one, replace = True, n_samples = 1998) 
#concatenate
df = pd.concat([zero, df_minority_upsampled])

from sklearn.utils import shuffle
df = shuffle(df) # shuffling so that there is particular sequence

df.Potability.value_counts().plot(kind ='pie');


# In[11]:


from sklearn.impute import SimpleImputer
imp= SimpleImputer(strategy= 'mean')
r= imp.fit_transform(df[['ph']])
s= imp.fit_transform(df[['Sulfate']])
t= imp.fit_transform(df[['Trihalomethanes']])

df['ph']=r
df['Sulfate']= s
df['Trihalomethanes']=t


# In[12]:


df.isnull().sum()


# In[13]:


df.head(4)


# In[40]:


plt.title('Missing Values Per Feature')
nans = df.isna().sum().sort_values(ascending=False).to_frame()
sns.heatmap(nans,annot=True,fmt='d',cmap='vlag')


# In[15]:


#Correlation

plt.figure(figsize=(10,8))
sns.set_context('paper')
sns.heatmap(df.corr(),cmap='Blues',linecolor='White',linewidth='1',annot=True,square=True)


# In[16]:


#EDA
non_potabale = df.query('Potability == 0')
potabale     = df.query('Potability == 1')

for ax,col in enumerate(df.columns[:9]):
    plt.subplot(3,3,ax+1)
    plt.title(f'Distribution of {col}')
    sns.kdeplot(x=non_potabale[col],label='Non Potabale')
    sns.kdeplot(x=potabale[col],label='Potabale')
    plt.legend(prop=dict(size=10))
    

plt.tight_layout()


# In[17]:


with pm.Model() as model:
    beta = pm.Normal("beta", mu=0, tau=0.001, testval=0)
    alpha = pm.Normal("alpha", mu=0, tau=1/df.Turbidity.std(), testval=0)
    p = pm.Deterministic("p_parm", 1.0/(1. + tt.exp(beta*df.Turbidity + alpha)))
    


# In[21]:


#Normalizing the data
df1=df
X = df1.iloc[:,:9].values
y = df1.iloc[:,9:10].values
#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print('Normalized data:')
print(X[0])


# In[22]:


#Splitting the model
#Train test split of model

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 5)


# #Splitting Data into train and test random.seed(1000) X=df1[['ph', 'Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Sulfate', 'Organic_carbon','Trihalomethanes','Turbidity']] # Features y=df1['Potability'] # Labels
# 
# Split dataset into training set and test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) # 70% training and 30% test

# In[24]:


#Statistical difference analysis
ttest_results_pvalues,ttest_results_statistic = [],[]
for ax,col in enumerate(df.columns[:9]):
    statistic,pvalue = ttest_ind(non_potabale[col],potabale[col])
    ttest_results_pvalues.append(pvalue)
    ttest_results_statistic.append(statistic)
    
ttest_res_df = pd.DataFrame({'S':ttest_results_statistic,'P':ttest_results_pvalues,'F':df.columns[:9]})
ttest_res_df = ttest_res_df.sort_values(by='P')


# In[25]:


tr  = go.Bar(x=ttest_res_df['F'] ,y=ttest_res_df['P'] ,name='T-test P Value')
tr2 = go.Bar(x=ttest_res_df['F'] ,y=ttest_res_df['S'] ,name='T-test F Statistic')

data = [tr2,tr]
fig = go.Figure(data=data,layout={'title':'T-test Results For Each Feature in Our Dataset','barmode':'overlay'})
fig.show()


# In[26]:


#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=1000)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)


# In[27]:


#After training, check the accuracy using actual and predicted values.
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[28]:


potability_feature = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Sulfate', 'Organic_carbon','Trihalomethanes','Turbidity']


# In[29]:


# checking the important features using Scikit - learn

feature_imp = pd.Series(clf.feature_importances_,index= potability_feature).sort_values(ascending=False)
feature_imp


# In[30]:


# we can visualize the importance as well
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[31]:


from xgboost import XGBClassifier


# #Splitting Data into train and test
# random.seed(1000)
# X2=df1[['ph', 'Hardness', 'Solids', 'Chloramines', 'Conductivity', 'Sulfate', 'Organic_carbon','Trihalomethanes','Turbidity']]  # Features
# y2=df1['Potability']  # Labels
# X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.3, random_state=1)

# In[32]:


model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')


# In[33]:


model.fit(X_train, y_train)


# In[34]:


# prdicting on test data
y_pred = model.predict(X_test)


# In[35]:


#Evaluating model performance
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[36]:


# checking the important features using Scikit - learn

feature_imp2 = pd.Series(model.feature_importances_,index= potability_feature).sort_values(ascending=False)
feature_imp2


# In[37]:


# we can visualize the importance as well
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Creating a bar plot
sns.barplot(x=feature_imp2, y=feature_imp2.index)
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# In[38]:


#xgboost gives a better accuracy of 67% and thevariables importance order changed.


# In[39]:


from sklearn.pipeline          import Pipeline
from sklearn.ensemble          import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.model_selection   import cross_val_score,train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.svm               import SVC

RandomForest_Pipeline     = Pipeline(steps = [('scale',StandardScaler()),('RF',RandomForestClassifier(random_state=42))])
AdaBoost_Pipeline         = Pipeline(steps = [('scale',StandardScaler()),('AB',AdaBoostClassifier(random_state=42))])
SVC_Pipeline              = Pipeline(steps = [('scale',StandardScaler()),('SVM',SVC(random_state=42))])


RandomForest_CV_f1     = cross_val_score(RandomForest_Pipeline,X_train,y_train,cv=10,scoring='f1')
AdaBoost_CV_f1         = cross_val_score(AdaBoost_Pipeline,X_train,y_train,cv=10,scoring='f1')
SVC_CV_f1              = cross_val_score(SVC_Pipeline,X_train,y_train,cv=10,scoring='f1')


# In[41]:


fig = make_subplots(rows=3, cols=1,shared_xaxes=True,subplot_titles=('Random Forest Cross Val Scores',
                                                                     'AdaBoost Cross Val Scores',
                                                                     'SVM Cross Val Scores'))

fig.add_trace(
    go.Scatter(x=np.arange(0,len(SVC_CV_f1)),y=RandomForest_CV_f1,name='Random Forest'),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(0,len(SVC_CV_f1)),y=AdaBoost_CV_f1,name='AdaBoost'),
    row=2, col=1
)
fig.add_trace(
    go.Scatter(x=np.arange(0,len(SVC_CV_f1)),y=SVC_CV_f1,name='SVM'),
    row=3, col=1
)

fig.update_layout(height=700, width=900, title_text="Different Baseline Models 10 Fold Cross Validation")
fig.update_yaxes(title_text="RMSE")
fig.update_xaxes(title_text="Fold #")
fig.show()


# In[42]:


pip install tune-sklearn ray[tune]


# In[43]:


#!pip install tune_sklearn
from tune_sklearn import TuneGridSearchCV


RFBE = RandomForestClassifier(random_state=42)
AdaBoost_Pipeline         = Pipeline(steps = [('scale',StandardScaler()),('AB',AdaBoostClassifier(random_state = 42,base_estimator = RFBE))])

AdaBoost_Pipeline.fit(X_train,y_train)

parameters = {'AB__base_estimator__max_depth':[2,3,5],
               'AB__base_estimator__min_samples_leaf':[2,5,10],
               'AB__base_estimator__criterion':['entropy','gini'],
               'AB__base_estimator__bootstrap':[True,False],
               'AB__n_estimators':[5,10,25],
               'AB__learning_rate':[0.01,0.1]}

ADA_RF_GS  = TuneGridSearchCV(AdaBoost_Pipeline,parameters,cv=3,verbose=1)
ADA_RF_GS  = GridSearchCV(AdaBoost_Pipeline,parameters,cv=3,verbose=10)
ADA_RF_GS.fit(X_train,y_train)

print("Best parameter (CV score=%0.3f):" % ADA_RF_GS.best_score_)
print(ADA_RF_GS.best_params_)


# In[44]:


RFBE = RandomForestClassifier(random_state=42,bootstrap=True,criterion='gini',max_depth=5,min_samples_leaf=10)
AdaBoost_Pipeline = Pipeline(steps = [('scale',StandardScaler()),('AB',AdaBoostClassifier(random_state = 42,
                                                                                                 base_estimator = RFBE,
                                                                                                 learning_rate=0.01,
                                                                                                 n_estimators=5))])

AdaBoost_Pipeline.fit(X_train,y_train)
f1 = AdaBoost_Pipeline.score(X_test,y_test)
print('F1 - Score of AdaBoost Model with Random Forest Base Estimators and Cross Validation Grid Search -[',np.round(f1,2),']')


# In[45]:


#Normalizing the data

X = df.iloc[:,:9].values
y = df.iloc[:,9:10].values
#Normalizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
print('Normalized data:')
print(X[0])


# In[46]:


#Splitting the model
#Train test split of model

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.1,random_state = 5)


# In[47]:


#Building and fitting the model

#importing libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# creating the model

model= keras.Sequential([
    layers.Dense(128, input_shape= (9,), activation= 'relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation= 'relu'),
    layers.Dropout(0.4),
    layers.Dense(32, activation= 'relu'),
    layers.Dropout(0.4),
    layers.Dense(1, activation= 'sigmoid')
    
])

model.compile(
    optimizer= 'adam',
    loss= 'binary_crossentropy',
    metrics= ['accuracy']
) 
history= model.fit(X_train, y_train, epochs=400,validation_data=(X_test, y_test), verbose= False)
model.evaluate(X_train, y_train)


# In[48]:


model.summary()


# In[49]:


model.evaluate(X_test, y_test)


# In[50]:


#Plotting accuracy and loss
# Model Accuracy

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[51]:


# Model Loss

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# In[52]:


model.save('water_potability_test.h5')


# In[53]:


#Model Predictions

from tensorflow.keras.models import load_model
m = load_model('water_potability_test.h5')
y_pred= model.predict(X_test)
y_pred = (y_pred>0.5)
y_pred[0:20]


# In[ ]:




