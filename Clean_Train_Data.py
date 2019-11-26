from datetime import date, datetime
import numpy as np
import pandas as pd
import datetime as dt

# Converts 'Date'-column to datetime format and creates extra columns:
# Year, Month, Week, Weekday, Day
def date_convert(df):
    '''
    Function converts the entries of a 'Date' column to datetime format and
    creates separate columns for: year, month, week, weekday, day.
    :param pandas.dataframe:
    :return pandas.datafram:
    '''
    if 'Date' in df.columns.tolist():
        df.loc[:, 'Date'] = pd.to_datetime(df['Date'])

        df['Year'] = df['Date'].dt.year
        df['Quarter'] = df['Date'].dt.quarter
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['WeekDay'] = df['Date'].dt.weekday
        df['Day'] = df['Date'].dt.day

        cols_after = df.columns.tolist()[0:9]
        cols_before = df.columns.tolist()[9:]
        df = df.loc[:, cols_before + cols_after]

    else:
        raise ValueError('Date column is not found in df')

    return df

# Converts CompetitionYear and CompetitionMonth to datetime format
def CompYear(df):
    '''
    Function combines 'CompetitionYear' and 'CompetitionMonth
    to new column 'CompetitionStart' in datetime format
    :param pandas.dataframe:
    :return pandas.dataframe:
    '''
    df['CompetitionStart'] = 'NaT'
    mask = (~df['CompetitionOpenSinceYear'].isnull()) & (~df['CompetitionOpenSinceMonth'].isnull())
    df['CompetitionStart'] = df.loc[mask, 'CompetitionOpenSinceYear'].astype(int).astype(str) \
                             + '-' \
                             + df.loc[mask, 'CompetitionOpenSinceMonth'].astype(int).astype(str) \
                             + '-01'
    df['CompetitionStart'] = pd.to_datetime(df['CompetitionStart'])
    return df

# Calculate if Competition is active on the Date and how long the competition is active
def CompAct(df):
    df['CompetitionActive'] = 0
    df['CompetitionDays'] = 0
    df.loc[df['CompetitionStart'] <= df['Date'], 'CompetitionActive'] = 1
    df['CompetitionDays'] = (df['Date'] - df['CompetitionStart'])/np.timedelta64(1, 'D')
    return df

# Create additional column: 'PromoDuration' -> how long is the Promotion running
def PromoDur(df):
    # Convert Promoyear and Promoweekno to datetime format
    df_subset = df.loc[(~df['Promo2SinceYear'].isnull()) & (~df['Promo2SinceWeek'].isnull()), \
                       ['Promo2SinceYear', 'Promo2SinceWeek']]
    df_subset = df_subset[['Promo2SinceYear', 'Promo2SinceWeek']].astype(int)
    df['PromoStart'] = df_subset.apply(lambda row: dt.datetime.strptime(f'{row.Promo2SinceYear} \
{row.Promo2SinceWeek} 1', '%G %V %u'), axis=1)

    # create PromoDuration Column:  Date - PromoStart
    df['PromoDuration'] = (df['Date'] - df['PromoStart'])/np.timedelta64(1, 'D')
    df['PromoDuration'].fillna(0, inplace=True)
    return df

# Create additional column: 'RunningAnyPromo'
# -> binary column representing if Promo1 or Promo2 is running on the Date
def RunAnyPromo(df):
    df['RunningAnyPromo'] = 0
    months_abbr = []

    for i in range(1, 13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & (df['Month'] == i[0]) & \
               (df['Promo2'] == 1) | df['Promo'] == 1
        df.loc[mask, 'RunningAnyPromo'] = 1
    return df

# Create additional column: 'RunPromo'
# -> binary column representing if Promo2 is running on the Date
def RunPromo(df):
    # Sets RunningPromo to 1 if Months in Substring of PromoIntervall and current month match
    df['RunningPromo2'] = 0
    months_abbr = []
    for i in range(1, 13):
        months_abbr.append((i, date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (df['PromoInterval'].str.contains(i[1], na=False)) & \
               (df['Month'] == i[0]) & (df['Promo2'] == 1)
        df.loc[mask, 'RunningPromo2'] = 1
    return df

# Data imputation of the 'Customer' column:
# if Store is open, Customer is set to mean(Customer) else 0
def CustImput(df):
    # Replace NaN in Customers with Mean(Customers), but if Store not open set Customers to 0
    df['Customers'].fillna(df['Customers'].mean(), inplace=True)
    df.loc[df['Open'] == 0, 'Customers'] = 0
    return df

# Create new columns: Rel (Relative Sales), ExpectedSales
def MeanSales(df, type='Train'):
    # Create new column: StoreInfo -> combination of StoreType and Assortment
    df['StoreInfo'] = df['Assortment'] + df['StoreType']
    df['Rel'] = np.nan
    df['ExpectedSales'] = np.nan
    if type == 'Train':
        # Calculate the mean Sales dependent on the StoreInfo normalized by the Customers
        mean_sales = df.loc[df.Sales > 0, ['Sales', 'Customers', 'StoreInfo']]\
            .groupby('StoreInfo').mean()
        mean_sales['Rel'] = mean_sales['Sales'] / mean_sales['Customers']
        b = mean_sales['Rel'].to_dict()
        df['Rel'] = df['StoreInfo'].map(b)
        mean_sales['Rel'].to_csv('traindata/MeanSales.csv', header=False)
        # calculate the expected sales from the sales normalized by the customers
        df['ExpectedSales'] = df['Customers'] * df['Rel']
        global_sales = np.mean(df['Sales'] / df['Customers'])
        with open('traindata/global_sales.txt', 'w') as f:
            f.write(str(global_sales))
    else:
        # in the case of the Test set, there will be no Sales column available
        # -> the relative sales are retrieved from the training set
        b = pd.read_csv('traindata/MeanSales.csv', header=None, names=['keys', 'values'])
        mydict = dict(zip(b['keys'], b['values']))
        df['Rel'] = df['StoreInfo'].map(mydict)
        df['ExpectedSales'] = df['Customers'] * df['Rel']
        with open('traindata/global_sales.txt') as f:
            global_sale = f.read()
        df.loc[df['Rel'].isnull(), 'ExpectedSales'] = float(global_sale)

    return df

# Root Mean Square Percentage Error  metric
def rmspe(actual: np.ndarray, predicted: np.ndarray):
    return np.sqrt(np.mean(np.square((actual - predicted) / actual)))

# Metric defined by Adam
def adam_metric(actual: np.ndarray, predicted: np.ndarray):
    return 100 * np.linalg.norm((actual - predicted) / actual) / np.sqrt(predicted.shape[0])

# one hot encoding
def onehotencoding(Train):
    cols = Train.select_dtypes(include='object').columns.tolist()
    Train = pd.get_dummies(Train, prefix=cols)
    return Train

if __name__ == "__main__":
    # Load data
    Store = pd.read_csv('data/store.csv', low_memory=False)
    Train = pd.read_csv('data/train.csv', low_memory=False)

    # Merge
    df = pd.merge(Store, Train, on='Store')
    df = date_convert(df)

    # Data Cleaning
    # Converts CompetitionYear and CompetitionMonth to datetime format
    df = CompYear(df)

    # Calculate if Competition is active on the Date and how long the competition is active
    df = CompAct(df)

    # Create additional column: 'PromoDuration' -> how long is the Promotion running
    df = PromoDur(df)

    # Create additional column: 'RunningAnyPromo'
    # -> binary column representing if Promo1 or Promo2 is running on the Date
    df = RunAnyPromo(df)

    # Create additional column: 'RunPromo'
    # -> binary column representing if Promo2 is running on the Date
    df = RunPromo(df)

    # Data imputation of the 'Customer' column:
    # if Store is open, Customer is set to mean(Customer) else 0
    df = CustImput(df)

    df.drop({'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',\
             'PromoStart', 'PromoInterval', 'Promo', 'Promo2', 'Promo2SinceYear', \
             'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
    df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', \
                                         'SchoolHoliday', 'CompetitionDistance'], inplace=True)
    df['CompetitionDays'].fillna(0, inplace=True)

    df = MeanSales(df, type='Train')    # Create new columns: Rel (Relative Sales), ExpectedSales
    df = onehotencoding(df)             # one hot encoding

    df.to_csv('data/CleanTrainData_ohe.csv')
