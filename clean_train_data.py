import numpy as np
import pandas as pd
import datetime as dt


def date_convert(dataframe):
    """
    Function converts the entries of a 'Date' column to datetime format and
    creates separate columns for: year, month, week, weekday, day.
    :param dataframe: Dataframe containing a 'Date' column
    :return dataframef: Dataframe with separated columns for: year, month, week, weekday, day
    """

    if 'Date' in dataframe.columns.tolist():
        dataframe.loc[:, 'Date'] = pd.to_datetime(dataframe['Date'])

        dataframe['Year'] = dataframe['Date'].dt.year
        dataframe['Quarter'] = dataframe['Date'].dt.quarter
        dataframe['Month'] = dataframe['Date'].dt.month
        dataframe['Week'] = dataframe['Date'].dt.week
        dataframe['WeekDay'] = dataframe['Date'].dt.weekday
        dataframe['Day'] = dataframe['Date'].dt.day

    else:
        raise ValueError('Date column is not found in df')

    return dataframe


def competition_start(dataframe):
    """
    Function combines 'CompetitionYear' and 'CompetitionMonth to new column 'CompetitionStart' in datetime format
    :param dataframe:
    :return dataframe: dataframe with new column 'CompetitionStart'
    """
    dataframe['CompetitionStart'] = 'NaT'
    mask = (~dataframe['CompetitionOpenSinceYear'].isnull()) & (~dataframe['CompetitionOpenSinceMonth'].isnull())
    dataframe['CompetitionStart'] = dataframe.loc[mask, 'CompetitionOpenSinceYear'].astype(int).astype(str) + '-' \
        + dataframe.loc[mask, 'CompetitionOpenSinceMonth'].astype(int).astype(str) + '-01'
    dataframe['CompetitionStart'] = pd.to_datetime(dataframe['CompetitionStart'])
    return dataframe


def competition_active(dataframe):
    """
    Determine if Competition is active on the given date and how long is competition already running
    :param dataframe:
    :return: dataframe with new columns 'CompetitionActive' and 'CompetitionDays'
    """
    dataframe['CompetitionActive'] = 0
    dataframe['CompetitionDays'] = 0
    dataframe.loc[dataframe['CompetitionStart'] <= dataframe['Date'], 'CompetitionActive'] = 1
    dataframe['CompetitionDays'] = (dataframe['Date'] - dataframe['CompetitionStart'])/np.timedelta64(1, 'D')
    return dataframe


def promo_duration(dataframe):
    """
    Determine how long the Promotion is already running. Creating new column 'PromoDuration'.
    :param dataframe:
    :return: dataframe with new column 'PromoDuration'
    """

    # Convert Promoyear and Promoweekno to datetime format
    df_subset = dataframe.loc[(~dataframe['Promo2SinceYear'].isnull()) & (~dataframe['Promo2SinceWeek'].isnull()),
                              ['Promo2SinceYear', 'Promo2SinceWeek']]
    df_subset = df_subset[['Promo2SinceYear', 'Promo2SinceWeek']].astype(int)
    dataframe['PromoStart'] = df_subset.apply(lambda row: dt.datetime.strptime(f'{row.Promo2SinceYear} \
        {row.Promo2SinceWeek} 1', '%G %V %u'), axis=1)

    # create PromoDuration Column:  Date - PromoStart
    dataframe['PromoDuration'] = (dataframe['Date'] - dataframe['PromoStart'])/np.timedelta64(1, 'D')
    dataframe['PromoDuration'].fillna(0, inplace=True)
    return dataframe


def run_any_promotion(dataframe):
    """
    Creates a new binary column 'RunningAnyPromo', which determines if Promo1 or Promo2 is runnning on the given date
    :param dataframe:
    :return: dataframe with new column 'RunningAnyPromo'
    """
    dataframe['RunningAnyPromo'] = 0
    months_abbr = []

    for i in range(1, 13):
        months_abbr.append((i, dt.date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (dataframe['PromoInterval'].str.contains(i[1], na=False)) & (dataframe['Month'] == i[0]) & \
               (dataframe['Promo2'] == 1) | dataframe['Promo'] == 1
        dataframe.loc[mask, 'RunningAnyPromo'] = 1
    return dataframe


def run_promo2(dataframe):
    """
    Create a new column 'RunPromo', which determines if Promo2 is running on the given date
    :param dataframe:
    :return: dataframe with new column 'RunPromo'
    """
    # Sets RunningPromo to 1 if Months in Substring of PromoIntervall and current month match
    dataframe['RunningPromo2'] = 0
    months_abbr = []
    for i in range(1, 13):
        months_abbr.append((i, dt.date(2008, i, 1).strftime('%b')))

    for i in months_abbr:
        mask = (dataframe['PromoInterval'].str.contains(i[1], na=False)) & (dataframe['Month'] == i[0]) & \
               (dataframe['Promo2'] == 1)
        dataframe.loc[mask, 'RunningPromo2'] = 1
    return dataframe


def customer_imputation(dataframe):
    """
    data imputation of the 'Customer' column: if store is open, customer is set to mean(customer) else 0
    :param dataframe:
    :return:
    """
    # Replace NaN in Customers with Mean(Customers), but if Store not open set Customers to 0
    dataframe['Customers'].fillna(dataframe['Customers'].mean(), inplace=True)
    dataframe.loc[dataframe['Open'] == 0, 'Customers'] = 0
    return dataframe


def store_info(dataframe):
    """
    create new columns 'StoreInfo', which combines Assortment and StoreType
    :param dataframe:
    :return:
    """
    dataframe['StoreInfo'] = dataframe['Assortment'] + dataframe['StoreType']
    return dataframe


def calculate_mean_sales(dataframe, data_type='Train'):
    """
    Create new columns: Rel (Relative Sales), ExpectedSales
    :param dataframe:
    :param data_type:
    :return:
    """
    dataframe['Rel'] = np.nan
    dataframe['ExpectedSales'] = np.nan
    if data_type == 'Train':
        # Calculate the mean Sales dependent on the StoreInfo normalized by the Customers
        mean_sales = dataframe.loc[dataframe.Sales > 0, ['Sales', 'Customers', 'StoreInfo']].groupby('StoreInfo').mean()
        mean_sales['Rel'] = mean_sales['Sales'] / mean_sales['Customers']
        b = mean_sales['Rel'].to_dict()
        dataframe['Rel'] = dataframe['StoreInfo'].map(b)
        mean_sales['Rel'].to_csv('traindata/MeanSales.csv', header=False)
        # calculate the expected sales from the sales normalized by the customers
        dataframe['ExpectedSales'] = dataframe['Customers'] * dataframe['Rel']
        global_sales = np.mean(dataframe['Sales'] / dataframe['Customers'])
        with open('traindata/global_sales.txt', 'w') as f:
            f.write(str(global_sales))
    else:
        # in the case of the Test set, there will be no Sales column available
        # -> the relative sales are retrieved from the training set
        b = pd.read_csv('traindata/MeanSales.csv', header=None, names=['keys', 'values'])
        mydict = dict(zip(b['keys'], b['values']))
        dataframe['Rel'] = dataframe['StoreInfo'].map(mydict)
        dataframe['ExpectedSales'] = dataframe['Customers'] * dataframe['Rel']
        with open('traindata/global_sales.txt') as f:
            global_sale = f.read()
        dataframe.loc[dataframe['Rel'].isnull(), 'ExpectedSales'] = float(global_sale)

    return dataframe


def rmspe(actual: np.ndarray, predicted: np.ndarray):
    """
    Root Mean Square Percentage Error  metric
    :param actual:
    :param predicted:
    :return:
    """
    return np.sqrt(np.mean(np.square((actual - predicted) / actual)))


def adam_metric(actual: np.ndarray, predicted: np.ndarray):
    """
    Metric defined by Adam
    :param actual:
    :param predicted:
    :return:
    """
    return 100 * np.linalg.norm((actual - predicted) / actual) / np.sqrt(predicted.shape[0])


def onehotencoding(train_df):
    """
    One-hot-Encoding of categorical columns
    :param train_df:
    :return:
    """
    cols = train_df.select_dtypes(include='object').columns.tolist()
    train_df = pd.get_dummies(train_df, prefix=cols)
    return train_df


def move_precition_variable(dataframe):
    """
    moves the 'Sale' column to the end of the dataframe
    :param dataframe:
    :return:
    """
    cols = list(dataframe.columns.values)
    cols.pop(cols.index('Sales'))
    dataframe = dataframe[cols + ['Sales']]
    return dataframe


if __name__ == "__main__":

    Store = pd.read_csv('data/store.csv', low_memory=False)
    Train = pd.read_csv('data/train.csv', low_memory=False)

    df = pd.merge(Store, Train, on='Store')
    df = date_convert(df)
    df = competition_start(df)
    df = competition_active(df)
    df = promo_duration(df)
    df = run_any_promotion(df)
    df = run_promo2(df)
    df = customer_imputation(df)

    df.drop({'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoStart', 'PromoInterval',
             'Promo', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
    df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance'],
              inplace=True)
    df['CompetitionDays'].fillna(0, inplace=True)
    df = store_info(df)
    df = calculate_mean_sales(df, data_type='Train')
    df = onehotencoding(df)
    df = move_precition_variable(df)

    df.to_csv('data/CleanTrainData_ohe.csv')
