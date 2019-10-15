import ast
import helper as hlp
import pandas as pd
import xgboost as xgb
import pickle

with open("traindata/params_ohe.txt", "r") as data:
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
test = hlp.onehotencoding(test)
#test.drop({'StoreType', 'Assortment','StateHoliday', 'StoreInfo'}, axis=1, inplace=True)

test.to_csv('data/CleanTestData_ohe.csv')

test = test.loc[test.Sales!= 0]
cols = list(test.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Sales')) #Remove sales from list
test = test[cols + ['Sales']] #Create new dataframe with sales right at the end
X, y = test.iloc[:, :-1],test.iloc[:, -1]

xgb_model = pickle.load(open("traindata/xgb_model_ohe.pickle.dat", "rb"))
df_DM = xgb.DMatrix(data=X, label=y)
params1 = dictionary
#xg_reg2 = xgb.XGBRegressor(**params1, n_estimators=500)
#xg_reg2.fit(X, y)
final_test_preds = xgb_model.predict(X)

#####calculate test statistics

EPSILON = 1e-10

print("RMSPE (test): %f" %(hlp.rmspe(y, final_test_preds)))
print("Adam's metric (test): %f" %(hlp.adam_metric(y, final_test_preds)))