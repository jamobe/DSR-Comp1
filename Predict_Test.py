import pickle
import pandas as pd
import helper as hlp

################ Loading the TEST Data ################################
TEST = pd.read_csv('data/TEST.csv', low_memory=False)
STORE = pd.read_csv('data/store.csv', low_memory=False)
TEST = pd.merge(STORE, TEST, on='Store')
TEST = hlp.date_convert(TEST)

################ Performing Data Cleaning and Manipulation ################################
TEST = hlp.CompYear(TEST)
TEST = hlp.CompAct(TEST)
TEST = hlp.PromoDur(TEST)
TEST = hlp.RunAnyPromo(TEST)
TEST = hlp.RunPromo(TEST)
TEST = hlp.CustImput(TEST)
TEST.drop({'Date', 'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', \
           'PromoStart', 'PromoInterval', 'Promo', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', \
           'DayOfWeek'}, axis=1, inplace=True)
TEST.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', \
                                       'SchoolHoliday', 'CompetitionDistance'], inplace=True)
TEST['CompetitionDays'].fillna(0, inplace=True)
TEST = hlp.MeanSales(TEST, type='TEST')
TEST = hlp.onehotencoding(TEST)

################ Write cleaned TEST set to a file ################################
TEST.to_csv('data/CleanTESTData_ohe.csv')

################# Separating the Sales column from the TEST set ################################
TEST = TEST.loc[TEST.Sales != 0]
COLS = list(TEST.columns.values) #Make a list of all of the columns in the df
COLS.pop(COLS.index('Sales')) #Remove sales from list
TEST = TEST[COLS + ['Sales']] #Create new dataframe with sales right at the end
X, Y = TEST.iloc[:, :-1], TEST.iloc[:, -1]

################# Load the trained model and predict TEST set ################################
XGB_MODEL = pickle.load(open("traindata/xgb_model_ohe.pickle.dat", "rb"))
TEST_PREDS = XGB_MODEL.predict(X)

################# Calculate TEST statistics ################################
print("RMSPE (TEST): %f" %(hlp.rmspe(Y, TEST_PREDS)))
print("Adam's metric (TEST): %f" %(hlp.adam_metric(Y, TEST_PREDS)))
