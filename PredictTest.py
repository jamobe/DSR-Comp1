import helper as hlp
import pandas as pd
import pickle

################ Loading the Test Data ################################
test = pd.read_csv('data/test.csv', low_memory=False)
store = pd.read_csv('data/store.csv', low_memory=False)
test = pd.merge(store, test, on='Store')
test = hlp.date_convert(test)

################ Performing Data Cleaning and Manipulation ################################
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

################ Write cleaned Test set to a file ################################
test.to_csv('data/CleanTestData_ohe.csv')

################# Separating the Sales column from the Test set ################################
test = test.loc[test.Sales!= 0]
cols = list(test.columns.values) #Make a list of all of the columns in the df
cols.pop(cols.index('Sales')) #Remove sales from list
test = test[cols + ['Sales']] #Create new dataframe with sales right at the end
X, y = test.iloc[:, :-1],test.iloc[:, -1]

################# Load the trained model and predict Test set ################################
xgb_model = pickle.load(open("traindata/xgb_model_ohe.pickle.dat", "rb"))
test_preds = xgb_model.predict(X)

################# Calculate test statistics ################################
print("RMSPE (test): %f" %(hlp.rmspe(y, test_preds)))
print("Adam's metric (test): %f" %(hlp.adam_metric(y, test_preds)))