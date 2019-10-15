from datetime import date, datetime
import numpy as np
import pandas as pd
import datetime as dt

# Creates extra Columns: Year, Month, Week, Weekday, Day
def date_convert(df):
    if 'Date' in df.columns.tolist():
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['WeekDay'] = df['Date'].dt.weekday
        df['Day'] = df['Date'].dt.day

        # df = df.drop(axis=1, labels='Date')

        cols_after = df.columns.tolist()[0:9]
        cols_before = df.columns.tolist()[9:]
        df = df.loc[:, cols_before + cols_after]

    else:
        raise ValueError('Date column is not found in df')

    return df

# Convert CompetitionYear and CompetitionMonth to datetime format
def CompYear(df):
    df['CompetitionStart'] = 'NaT'
    mask = (~df['CompetitionOpenSinceYear'].isnull()) & (~df['CompetitionOpenSinceMonth'].isnull())
    df['CompetitionStart'] = df.loc[mask, 'CompetitionOpenSinceYear'].astype(int).astype(str) + '-' + df.loc[mask, 'CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01'
    df['CompetitionStart'] = pd.to_datetime(df['CompetitionStart'])
    return df

# Calculate is Competition is active and how long the competition is active
def CompAct(df):
    df['CompetitionActive'] = 0
    df['CompetitionDays'] = 0
    df.loc[df['CompetitionStart'] <= df['Date'], 'CompetitionActive'] = 1
    df['CompetitionDays'] = (df['Date'] - df['CompetitionStart'])/np.timedelta64(1,'D')
    return df

def PromoDur(df):
    # Convert Promoyear and Promoweekno to datetime format
    df_subset = df.loc[(~df['Promo2SinceYear'].isnull()) & (~df['Promo2SinceWeek'].isnull()), ['Promo2SinceYear','Promo2SinceWeek']]
    df_subset = df_subset[['Promo2SinceYear','Promo2SinceWeek']].astype(int)
    df['PromoStart'] = df_subset.apply(lambda row: dt.datetime.strptime(f'{row.Promo2SinceYear} {row.Promo2SinceWeek} 1', '%G %V %u'), axis=1)

    # create PromoDuration Column:  Date - PromoStart
    df['PromoDuration'] = (df['Date'] - df['PromoStart'])/np.timedelta64(1,'D')
    df['PromoDuration'].fillna(0, inplace=True)
    return df

# Create RunnningAnyPromo Column
def RunAnyPromo(df):
    df['RunningAnyPromo'] = 0
    months_abbr = []

    for i in range(1,13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & (df['Month']==i[0]) & (df['Promo2']==1) | df['Promo']==1
        df.loc[mask, 'RunningAnyPromo'] = 1
    return df

def RunPromo(df):
    # Sets RunningPromo to 1 if Months in Substring of PromoIntervall and current month match
    df['RunningPromo2'] = 0
    months_abbr = []
    for i in range(1, 13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & (df['Month'] == i[0]) & (df['Promo2'] == 1)
        df.loc[mask, 'RunningPromo2'] = 1
    return df

def CustImput(df):
    # Replace NaN in Customers with Mean(Customers), but if Store not open set Customers to 0
    df['Customers'].fillna(df['Customers'].mean(), inplace=True)
    df.loc[df['Open'] == 0, 'Customers'] = 0
    return df

# calculate mean sales per number of customers per each store type
def MeanSales(df, type='Train'):
    df['StoreInfo'] = df['Assortment'] + df['StoreType']
    df['Rel'] = np.nan
    df['ExpectedSales'] = np.nan
    if type == 'Train':
        mean_sales = df.loc[df.Sales > 0, ['Sales', 'Customers', 'StoreInfo']].groupby('StoreInfo').mean()
        mean_sales['Rel'] = mean_sales['Sales'] / mean_sales['Customers']
        b = mean_sales['Rel'].to_dict()
        df['Rel'] = df['StoreInfo'].map(b)
        mean_sales['Rel'].to_csv('traindata/MeanSales.csv', header=False)
        df['ExpectedSales'] = df['Customers'] * df['Rel']
        global_sales = np.mean(df['Sales'] / df['Customers'])
        with open('traindata/global_sales.txt', 'w') as f:
            f.write(str(global_sales))
    else:
        b = pd.read_csv('traindata/MeanSales.csv', header=None, names=['keys', 'values'])
        mydict = dict(zip(b['keys'], b['values']))
        df['Rel'] = df['StoreInfo'].map(mydict)
        df['ExpectedSales'] = df['Customers'] * df['Rel']
        with open('traindata/global_sales.txt') as f:
            global_sale = f.read()
        df.loc[df['Rel'].isnull(), 'ExpectedSales'] = float(global_sale)

    return df

def rmspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean(np.square((actual - predicted) / actual)))

def adam_metric(actual: np.ndarray, predicted: np.ndarray):
    #preds = predicted.reshape(-1)
    #actuals = actual.reshape(-1)
    #assert predicted.shape == actuals.shape
    return 100 * np.linalg.norm((actual - predicted) / actual) / np.sqrt(predicted.shape[0])

'''def one_hot_enc(Train):
    # OHE

    cols = Train.select_dtypes(include='object').columns.tolist()
    for col in cols:
        Train[col] = Train[col].astype(str)
    ohe = OneHotEncoder(handle_unknown='ignore')

    ohe_train = pd.DataFrame()

    for col in cols:
        ohe.fit(np.array(Train[col]).reshape(-1, 1))
        pickle.dump(ohe, open('traindata/ohe_' + col, "wb"))
        ohe_train_tmp = pd.DataFrame(columns=ohe.categories_, data=ohe.transform(np.array(Train[col]).reshape(-1, 1)).toarray())
        ohe_train = pd.concat([ohe_train, ohe_train_tmp], axis=1)

    # Drop columns
    #Train.drop(axis=1, labels=cols, inplace=True)

    # Concat
    Train = pd.concat([Train.reset_index(), ohe_train], axis=1)

    return Train


def one_hot_enc_test(Test):
    # OHE
    cols = Test.select_dtypes(include='object').columns.tolist()
    for col in cols:
        Test[col] = Test[col].astype(str)

    ohe_test = pd.DataFrame()

    for col in cols:
        ohe = pickle.load(open('traindata/ohe_' + col, "rb"))
        ohe_test_tmp = pd.DataFrame(columns=ohe.categories_, data=ohe.transform(np.array(Test[col]).reshape(-1, 1)).toarray())
        ohe_test = pd.concat([ohe_test, ohe_test_tmp], axis=1)

    # Drop columns
    #Test.drop(axis=1, labels=cols, inplace=True)

    # Concat
    Test = pd.concat([Test.reset_index(), ohe_test], axis=1)
    return Test'''