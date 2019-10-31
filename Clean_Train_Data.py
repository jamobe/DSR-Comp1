import pandas as pd
import helper as hlp

# Load data
Store = pd.read_csv('data/store.csv', low_memory=False)
Train = pd.read_csv('data/train.csv', low_memory=False)

# Merge
df = pd.merge(Store, Train, on='Store')
df = hlp.date_convert(df)

# Data Cleaning
# Converts CompetitionYear and CompetitionMonth to datetime format
df = hlp.CompYear(df)

# Calculate if Competition is active on the Date and how long the competition is active
df = hlp.CompAct(df)

# Create additional column: 'PromoDuration' -> how long is the Promotion running
df = hlp.PromoDur(df)

# Create additional column: 'RunningAnyPromo'
# -> binary column representing if Promo1 or Promo2 is running on the Date
df = hlp.RunAnyPromo(df)

# Create additional column: 'RunPromo'
# -> binary column representing if Promo2 is running on the Date
df = hlp.RunPromo(df)

# Data imputation of the 'Customer' column:
# if Store is open, Customer is set to mean(Customer) else 0
df = hlp.CustImput(df)

df.drop({'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',\
         'PromoStart', 'PromoInterval', 'Promo', 'Promo2', 'Promo2SinceYear', \
         'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', \
                                     'SchoolHoliday', 'CompetitionDistance'], inplace=True)
df['CompetitionDays'].fillna(0, inplace=True)

df = hlp.MeanSales(df, type='Train')    # Create new columns: Rel (Relative Sales), ExpectedSales
df = hlp.onehotencoding(df)             # one hot encoding

df.to_csv('data/CleanTrainData_ohe.csv')
