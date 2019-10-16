import helper as hlp
import pandas as pd
import xgboost as xgb
from datetime import timedelta
from bayes_opt import BayesianOptimization
import pickle

################# Loading the cleaned Training Data ################################
df = pd.read_csv('data/CleanTrainData_ohe.csv',index_col=0)
df = df.loc[df.Sales != 0]

################# Splitting the data in Train and Validation Sets ################################
# Separate the 'Sale'-Column from the remaining Columns
df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
cols = list(df.columns.values)
cols.pop(cols.index('Sales'))
df = df[cols + ['Sales']]
X, y = df.iloc[:, :-1],df.iloc[:, -1]

# Split the data into Training and Validation Data based on the time
date_range_days=(df.Date.max() - df.Date.min()).days
split_date=df.Date.min() + timedelta(date_range_days*0.8)
df_early,df_later = df.loc[df.Date <= split_date], df.loc[df.Date > split_date]

################# Remove Datetime-Format Columns ################################
# Remove Date-Column, since Datetime-Format is not accepted by XGBoost
df_early.drop({'Date'}, axis=1, inplace=True)
df_later.drop({'Date'}, axis=1, inplace=True)

################# Create X-FeatureMatrix and Y-PredictorVector ################################
X_train, X_val, y_train, y_val = df_early.iloc[:,:-1], df_later.iloc[:,:-1], df_early.iloc[:,-1], df_later.iloc[:,-1]

################# Defining the hyper parameter ################################
params = {
# Learning Task Parameters
    'objective': 'reg:squarederror',
    'eval_metric':'rmse', # Evaluation metrics for validation data
# Parameters for Tree Booster
    'learning_rate': 0.05, # Learning Rate: step size shrinkage used to prevent overfitting.
# Paramters for XGB ScikitLearn API
    'n_jobs': 4, # Number of parallel threads used to run xgboost
    'n_estimators': 1000, # number of trees you want to build
    'verbosity': 2, # degree of verbosity: 0 (silent) - 3 (debug)
    'max_depth': 4,
    'reg_lambda': 1,
    'colsample_bytree': 0.5,
    'subsample': 0.9
}

################# Defining the fit parameters ################################
fit_params = {
            'eval_metric':'rmse',
            'early_stopping_rounds': 10,
            'eval_set': [(X_val, y_val)],
            }

################# Initiate XGB regressor and run initial fit ################################
xgb_reg = xgb.XGBRegressor(**params)
xgb_reg.fit(X_train, y_train, **fit_params)

################# Create XGB DMatrix for xgb.cv Function ################################
df_DM = xgb.DMatrix(data=X_train, label=y_train)

################# Creating an evaluation function to be optimized ################################
def xgb_evaluate(max_depth, reg_lambda, colsample_bytree, subsample):
    params1 = {
        'colsample_bytree': colsample_bytree,
        'max_depth': int(round(max_depth)),  # Maximum depth of a tree: high value -> prone to overfitting
        'reg_lambda': reg_lambda,  # L2 regularization term on weights
        'subsample': subsample
    }
    cv_result = xgb.cv(dtrain=df_DM,
                        params=params1,
                        early_stopping_rounds=10,
                        num_boost_round=100,
                        metrics='rmse')
    return -cv_result['test-rmse-mean'].iloc[-1]

################# Using BayesOptimization for evaluation function ################################
optimizer = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 5),
                                                    'reg_lambda': (0, 5),
                                                    'colsample_bytree': (0.3, 0.8),
                                                    'subsample': (0.8, 1)})

################# calculate maximum of Bayes optimization ################################
optimizer.maximize(init_points=10, n_iter=10)
# n_iter: How many steps of bayesian optimization you want to perform. The more steps the more likely to find a good maximum you are.
# init_points: How many steps of random exploration you want to perform. Random exploration can help by diversifying the exploration space.

################# retrieving optimized parameters ################################
params1 = optimizer.max['params']
params1['max_depth']= int(round(params1['max_depth']))
params.update(params1)

################# Run model with optimized parameters ################################
xgb_reg = xgb.XGBRegressor(**params)
xgb_reg.fit(X_train, y_train, **fit_params)

################# prediction for Train and Validation set ################################
train_preds = xgb_reg.predict(X_train)
val_preds = xgb_reg.predict(X_val)

################# Print metrics for Train and Validation set prediction ################################
print("RMSE train: %f" % hlp.rmspe(y_train, train_preds))
print("RMSE CV: %f" % hlp.rmspe(y_val, val_preds))
print("Adam's metric (train): %f" %(hlp.adam_metric(y_train, train_preds)))
print("Adam's metric (CV): %f" %(hlp.adam_metric(y_val, val_preds)))

################# save model to file  ################################
pickle.dump(xgb_reg, open("traindata/xgb_model_ohe.pickle.dat", "wb"))
