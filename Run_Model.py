from datetime import timedelta
import pickle
import pandas as pd
import clean_train_data as hlp
from bayes_opt import BayesianOptimization
import xgboost as xgb


def split_training_data(dataframe):
    """
    Splitting the training data into train and validation set considering the dates
    :param dataframe:
    :return:
    """
    date_range_days = (dataframe.Date.max() - dataframe.Date.min()).days
    split_date = dataframe.Date.min() + timedelta(date_range_days * 0.8)
    df_early, df_late = dataframe.loc[dataframe.Date <= split_date], dataframe.loc[dataframe.Date > split_date]
    df_early.drop({'Date'}, axis=1, inplace=True)
    df_late.drop({'Date'}, axis=1, inplace=True)
    x_train, x_val, y_train, y_val = df_early.iloc[:, :-1], df_late.iloc[:, :-1], df_early.iloc[:, -1], \
        df_late.iloc[:, -1]
    return x_train, x_val, y_train, y_val


def xgb_evaluate(max_depth, reg_lambda, colsample_bytree, subsample):
    params1 = {
        'colsample_bytree': colsample_bytree,
        'max_depth': int(round(max_depth)),  # Maximum depth of a tree: high value
        'reg_lambda': reg_lambda,  # L2 regularization term on weights
        'subsample': subsample
    }
    cv_result = xgb.cv(dtrain=df_DM, params=params1, early_stopping_rounds=10, num_boost_round=100, metrics='rmse')
    return -cv_result['test-rmse-mean'].iloc[-1]


if __name__ == "__main__":
    df = pd.read_csv('data/CleanTrainData_ohe.csv', index_col=0)
    df = df.loc[df.Sales != 0]
    df.loc[:, 'Date'] = pd.to_datetime(df['Date'])
    X_train, X_val, Y_train, Y_val = split_training_data(df)

    params = {
        # Learning Task Parameters
        'objective': 'reg:squarederror',
        # Parameters for Tree Booster
        'learning_rate': 0.05,  # Learning Rate: step size shrinkage used to prevent overfitting.
        # Paramters for XGB ScikitLearn API
        'n_jobs': 4,  # Number of parallel threads used to run xgboost
        'n_estimators': 1000,  # number of trees you want to build
        'verbosity': 2,  # degree of verbosity: 0 (silent) - 3 (debug)
        'max_depth': 4,
        'reg_lambda': 1,
        'colsample_bytree': 0.5,
        'subsample': 0.9
    }

    fit_params = {
        'eval_metric': 'rmse',
        'early_stopping_rounds': 10,
        'eval_set': [(X_val, Y_val)],
    }

    xgb_reg = xgb.XGBRegressor(**params)
    df_DM = xgb.DMatrix(data=X_train, label=Y_train)
    optimizer = BayesianOptimization(xgb_evaluate, {'max_depth': (3, 5),
                                                    'reg_lambda': (0, 5),
                                                    'colsample_bytree': (0.3, 0.8),
                                                    'subsample': (0.8, 1)})

    optimizer.maximize(init_points=10, n_iter=10)
    params_1 = optimizer.max['params']
    params_1['max_depth'] = int(round(params_1['max_depth']))
    params.update(params_1)
    xgb_reg.fit(X_train, Y_train, **fit_params)

    train_preds = xgb_reg.predict(X_train)
    val_preds = xgb_reg.predict(X_val)

    print("RMSE train: %f" % hlp.rmspe(Y_train, train_preds))
    print("RMSE CV: %f" % hlp.rmspe(Y_val, val_preds))
    print("Adam's metric (train): %f" % hlp.adam_metric(Y_train, train_preds))
    print("Adam's metric (CV): %f" % hlp.adam_metric(Y_val, val_preds))

    pickle.dump(xgb_reg, open("traindata/xgb_model_ohe.pickle.dat", "wb"))
