import helper as hlp
import pandas as pd
import numpy as np

# Load data
store = pd.read_csv('data/store.csv')
train = pd.read_csv('data/train.csv', dtype = {'StateHoliday': np.str})

# Merge
train = pd.merge(store, train, on='Store')
train = hlp.date_convert(train)

# Split
end_val = np.floor(0.8 * train.shape[0]).astype(int)
end_test = np.floor(0.9 * train.shape[0]).astype(int)

Train = train.loc[:end_val]
Val = train.loc[end_val:end_test]
Test = train.loc[end_test:]

dflist = [Train, Val, Test]
for df in dflist:
    # Data Cleaning
    df = hlp.CompYear(df)
    df = hlp.CompAct(df)
    df = hlp.PromoDur(df)
    df = hlp.RunAnyPromo(df)
    df = hlp.RunPromo(df)
    df = hlp.CustImput(df)

    # Remove Columns
    df.drop({'Date','CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoStart','PromoInterval','Promo','Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'DayOfWeek'}, axis=1, errors='ignore', inplace=True)

    # Remove empty Rows
    df = df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', 'SchoolHoliday','CompetitionDistance'], inplace=True)

    # Replace NaN by 0's
    df['CompetitionDays'].fillna(0, inplace=True)

Train = hlp.MeanSales(Train, type='Train')
Val = hlp.MeanSales(Val, type='Val')
Test = hlp.MeanSales(Test, type='Test')

Train.to_csv('CheckTrainData.csv')
Val.to_csv('CheckValData.csv')
Test.to_csv('CheckTestData.csv')