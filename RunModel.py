
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split as train_test_split
from datetime import timedelta
import datetime as dt
from bayes_opt import BayesianOptimization
import pickle

df = pd.read_csv('data/CleanTrainData.csv',index_col=0)
df = df.loc[df.Sales!= 0]

# Create X_train, y_train and so on
# XGB
df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
cols = list(df.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Sales')) #Remove sales from list
df = df[cols + ['Sales']] #Create new dataframe with sales right at the end
X, y = df.iloc[:, :-1],df.iloc[:, -1]

date_range_days=(df.Date.max() - df.Date.min()).days
split_date=df.Date.min() + timedelta(date_range_days*0.8) #train set 80% of full population
#randomly creating train and test subsets. may need to refine this
df_early,df_later = df.loc[df.Date <= split_date], df.loc[df.Date > split_date]
#create feature matrix of everything up to sales, create labels from sales
df_early.drop({'Date'}, axis=1, inplace=True)
df_later.drop({'Date'}, axis=1, inplace=True)
X_train, X_val, y_train, y_val = df_early.iloc[:,:-1], df_later.iloc[:,:-1], df_early.iloc[:,-1], df_later.iloc[:,-1]

# creating XGB optimised data structure. we will need this for our cross validation model later
df_DM = xgb.DMatrix(data=X_train, label=y_train)

params = {"objective":"reg:squarederror", #type of regressor, shouldnt change
            'silent': False,
            'colsample_bytree': 0.627, #percentage of features used per tree. High value can lead to overfitting.
            'learning_rate': 0.05, #step size shrinkage used to prevent overfitting. Range is [0,1]
            'max_depth': 5, #determines how deeply each tree is allowed to grow during any boosting round. keep this low! this will blow up our variance if high
            'lambda': 4.655, #L1 regularization on leaf weights. A large valupythone leads to more regularization. Could consider l2 euclidiean regularisation
            'n_estimators': 1250, #number of trees you want to build. 1250
            'n_jobs': 4,#should optimise core usage on pc
            'subsample':0.86}

#now we must instantiate the XGB regressor by calling XGB regressor CLASS from the XGBoost library, we must give it the hypter parameters as arguments
xg_reg = xgb.XGBRegressor(**params)

#Fit the regressor to the training set and make predictions for the test set using .fit() and .predict() methods
xg_reg.fit(X_train, y_train)
preds = xg_reg.predict(X_val)
preds_train = xg_reg.predict(X_train)

def xgb_evaluate(max_depth, lambd, colsample_bytree, subsample):
    params1 = {'objective': 'reg:squarederror',
               'silent': False,
               'max_depth': int(max_depth),
               'learning_rate': 0.05,
               'lambda': lambd,
               'subsample': subsample,
               'colsample_bytree': colsample_bytree,
               #'n_estimators': 100,
               'n_jobs': 4}
    # Used around 1000 boosting rounds in the full model
    cv_result = xgb.cv(dtrain=df_DM, params=params1, num_boost_round=125, nfold=3, metrics='rmse', seed=42)
    # Bayesian optimization only knows how to maximize, not minimize, so return the negative RMSE
    return -1.0 * cv_result['test-rmse-mean'].iloc[-1]

xgb_bo = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 5),
                                                 'lambd': (0, 5),
                                                 'colsample_bytree': (0.3, 0.8),
                                                'subsample': (0.8,1)})

# Use the expected improvement acquisition function to handle negative numbers
# Optimally needs quite a few more initiation points and number of iterations
xgb_bo.maximize(init_points=10, n_iter=3, acq='ei') #init_points=10
#extract best parameters from model
params1 = xgb_bo.max['params']
print (params1)
#Converting the max_depth and from float to int
params1['max_depth']= int(params1['max_depth'])

xg_reg2 = xgb.XGBRegressor(**params1,n_estimators=500)
xg_reg2.fit(X_train, y_train)
train_preds1 = xg_reg2.predict(X_train)
val_preds1 = xg_reg2.predict(X_val)

EPSILON = 1e-10

def rmspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean(np.square((actual - predicted) / actual)))

def adam_metric(actual: np.ndarray, predicted: np.ndarray):
    #preds = predicted.reshape(-1)
    #actuals = actual.reshape(-1)
    #assert predicted.shape == actuals.shape
    return 100 * np.linalg.norm((actual - predicted) / (actual)) / np.sqrt(predicted.shape[0])

print("RMSE train: %f" % rmspe(y_train, train_preds1))
print("RMSE CV: %f" % rmspe(y_val, val_preds1))
print("Adam's metric (train): %f" %(adam_metric(y_train, train_preds1)))
print("Adam's metric (CV): %f" %(adam_metric(y_val, val_preds1)))

with open('traindata/params.txt','w') as f:
    f.write(str(params1))
    f.close()

# save model to file
pickle.dump(xg_reg2, open("traindata/xgb_model.pickle.dat", "wb"))