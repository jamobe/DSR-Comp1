import helper as hlp
import pandas as pd
import numpy as np

# Load data
store = pd.read_csv('data/store.csv', low_memory=False)
train = pd.read_csv('data/train.csv', low_memory=False)

# Merge
df = pd.merge(store, train, on='Store')
df = hlp.date_convert(df)

# Data Cleaning
df = hlp.CompYear(df)
df = hlp.CompAct(df)
df = hlp.PromoDur(df)
df = hlp.RunAnyPromo(df)
df = hlp.RunPromo(df)
df = hlp.CustImput(df)
df.drop({'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoStart','PromoInterval','Promo','Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', 'SchoolHoliday','CompetitionDistance'], inplace=True)
df['CompetitionDays'].fillna(0, inplace=True)

df = hlp.MeanSales(df, type='Train')
df.drop({'StoreType', 'Assortment','StateHoliday', 'StoreInfo'}, axis=1, inplace=True)

df.to_csv('data/CleanTrainData.csv')
