import pandas as pd
import helper as hlp

# Load data
STORE = pd.read_csv('data/store.csv', low_memory=False)
TRAIN = pd.read_csv('data/train.csv', low_memory=False)

# Merge
DF = pd.merge(STORE, TRAIN, on='Store')
DF = hlp.date_convert(DF)

# Data Cleaning
# Converts CompetitionYear and CompetitionMonth to datetime format
DF = hlp.CompYear(DF)

# Calculate if Competition is active on the Date and how long the competition is active
DF = hlp.CompAct(DF)

# Create additional column: 'PromoDuration' -> how long is the Promotion running
DF = hlp.PromoDur(DF)

# Create additional column: 'RunningAnyPromo'
# -> binary column representing if Promo1 or Promo2 is running on the Date
DF = hlp.RunAnyPromo(DF)

# Create additional column: 'RunPromo'
# -> binary column representing if Promo2 is running on the Date
DF = hlp.RunPromo(DF)

# Data imputation of the 'Customer' column:
# if Store is open, Customer is set to mean(Customer) else 0
DF = hlp.CustImput(DF)

DF.drop({'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth',\
         'PromoStart', 'PromoInterval', 'Promo', 'Promo2', 'Promo2SinceYear', \
         'Promo2SinceWeek', 'DayOfWeek'}, axis=1, inplace=True)
DF.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', \
                                     'SchoolHoliday', 'CompetitionDistance'], inplace=True)
DF['CompetitionDays'].fillna(0, inplace=True)

DF = hlp.MeanSales(DF, type='Train')    # Create new columns: Rel (Relative Sales), ExpectedSales
DF = hlp.onehotencoding(DF)             # one hot encoding

DF.to_csv('data/CleanTrainData_ohe.csv')
