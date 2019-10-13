import ast
import helper as hlp
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

with open("traindata/params.txt", "r") as data:
    dictionary = ast.literal_eval(data.read())

test = pd.read_csv('data/test.csv', low_memory=False)
store = pd.read_csv('data/store.csv', low_memory=False)

test = pd.merge(store, test, on='Store')
test = hlp.date_convert(test)

test = hlp.CompYear(test)
test = hlp.CompAct(test)
test = hlp.PromoDur(test)
test = hlp.RunAnyPromo(test)
test = hlp.RunPromo(test)
test = hlp.CustImput(test)
test.drop({'Date', 'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoStart','PromoInterval','Promo','Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
test.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', 'SchoolHoliday','CompetitionDistance'], inplace=True)
test['CompetitionDays'].fillna(0, inplace=True)
test = hlp.MeanSales(test, type='Test')

test.to_csv('data/CleanTestData.csv')

cols = list(test.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Sales')) #Remove sales from list
test = test[cols + ['Sales']] #Create new dataframe with sales right at the end
X, y = test.iloc[:, :-1],test.iloc[:, -1]

#df_DM = xgb.DMatrix(data=X, label=y)
params1 = dictionary
#xg_reg2 = xgb.XGBRegressor(**params1, n_estimators=500)
#xg_reg2.fit(X, y)
xgb = pickle.load(open("traindata/xgb_model.pickle.dat", "rb"))
final_test_preds = xgb.predict(X)

#####calculate test statistics


EPSILON = 1e-10

def rmspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean(np.square((actual - predicted) / (actual + EPSILON))))

def adam_metric(actuals, preds):
    #preds = preds.reshape(-1)
    #actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / (actuals + 1e-9)) / np.sqrt(preds.shape[0])

print("RMSPE (test): %f" %(rmspe(y, final_test_preds)))
print("Adam's metric (test): %f" %(adam_metric(y, final_test_preds)))