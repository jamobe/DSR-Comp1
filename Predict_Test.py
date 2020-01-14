import pickle
import pandas as pd
import clean_train_data as hlp


if __name__ == "__main__":

    Test = pd.read_csv('data/test.csv', low_memory=False)
    Store = pd.read_csv('data/store.csv', low_memory=False)

    test_df = pd.merge(Store, Test, on='Store')
    test_df = hlp.date_convert(test_df)
    test_df = hlp.competition_start(test_df)
    test_df = hlp.competition_active(test_df)
    test_df = hlp.promo_duration(test_df)
    test_df = hlp.run_any_promotion(test_df)
    test_df = hlp.run_promo2(test_df)
    test_df = hlp.customer_imputation(test_df)
    test_df.drop({'Date', 'CompetitionStart', 'CompetitionOpenSinceYear', 'CompetitionOpenSinceMonth', 'PromoStart',
                  'PromoInterval', 'Promo', 'Promo2', 'Promo2SinceYear', 'Promo2SinceWeek', 'DayOfWeek'},
                 axis=1, inplace=True)
    test_df.dropna(axis=0, how='any', subset=['Sales', 'Open', 'StateHoliday', 'SchoolHoliday', 'CompetitionDistance'],
                   inplace=True)
    test_df['CompetitionDays'].fillna(0, inplace=True)
    test_df = hlp.store_info(test_df)
    test_df = hlp.calculate_mean_sales(test_df, data_type='test')
    test_df = hlp.onehotencoding(test_df)
    test_df = hlp.move_precition_variable(test_df)
    test_df.to_csv('data/CleantestData_ohe.csv')
    test_df = test_df.loc[test_df.Sales != 0]
    X, Y = test_df.iloc[:, :-1], test_df.iloc[:, -1]

    XGB_MODEL = pickle.load(open("traindata/xgb_model_ohe.pickle.dat", "rb"))
    test_preds = XGB_MODEL.predict(X)

    print("RMSPE (test): %f" % hlp.rmspe(Y, test_preds))
    print("Adam's metric (test): %f" % hlp.adam_metric(Y, test_preds))
