from __future__ import division
import numpy as np
import pandas as pd
import os
from keras.models import load_model
from datetime import datetime, timedelta
import sys
import json


# Let's use time component and transform data with log
def log_scale_dataframe(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = np.log(df[col] / df[col].shift() )
    return new_df


# Prepare dataset  - take events for country
def split_events(gdelt_df, country_code, day0,  day1, day2, day3, day4):
    event_root_codes = gdelt_df.QuadClass.unique()
    measure_name = 'NumArticles'
    event_series = [gdelt_df[(gdelt_df.QuadClass == event_code) & (gdelt_df.Actor1Geo_CountryCode == country_code)  & ((gdelt_df.Date == day0) | (gdelt_df.Date == day1) | (gdelt_df.Date == day2) | (gdelt_df.Date == day3) | (gdelt_df.Date == day4) )][[measure_name]] for event_code in event_root_codes]

    event_by_codes = pd.concat(event_series, axis=1).sort_index()
    event_by_codes.columns = map(str, event_root_codes)
    event_by_codes = event_by_codes.fillna(method='ffill')
    event_by_codes = event_by_codes.fillna(method='bfill')
    return event_by_codes


def append_previous_timestamps(X):

    # This function appends previous two timestamps
    feature = []
    i = 0
    j = 1
    k = 2

    features = [X[k][0], X[k][1], X[k][2], X[k][3], X[j][0], X[j][1], X[j][2], X[j][3], X[i][0], X[i][1], X[i][2],
                X[i][3]]

    feature.append(features)
    feature = np.array(feature)

    return feature


def main():

    # PARAMETERS
    input_file =  r'C:\Users\lenovo\PycharmProjects\FYP\lstm\reduced.csv'
    # country_code = 'PK'
    # day4 = '2019-02-28'
    # predict_for = 4

    predict_for = int(sys.argv[1])
    country_code = sys.argv[2]
    day4 = sys.argv[3]

    '''
    1: Material Cooperation
    2: Verbal Cooperation
    3: Verbal Conflict
    4: Material Conflict
    '''

    file_names = {
        1: 'materialcooperation',
        2: 'verbalcooperation',
        3: 'verbalconflict',
        4: 'materialconflict'
    }

    date = datetime.strptime(day4, "%Y-%m-%d")

    day4 = date.strftime("%Y-%m-%d")
    day3 = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    day2 = (date - timedelta(days=2)).strftime("%Y-%m-%d")
    day1 = (date - timedelta(days=3)).strftime("%Y-%m-%d")
    day0 = (date - timedelta(days=4)).strftime("%Y-%m-%d")

    pd.options.display.float_format = '{:.2f}'.format

    gdelt_df = pd.read_csv(input_file)

    gdelt_df['SQLDATE']=gdelt_df['SQLDATE'].apply(str)
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'])
    gdelt_df.index = gdelt_df['Date']
    gdelt_df.drop(['SQLDATE'], axis=1, inplace=True)

    event_by_codes = split_events(gdelt_df, country_code, day0, day1, day2, day3, day4)
    event_by_codes_log = log_scale_dataframe(event_by_codes)

    # Prepare X (all quads classes' Growth Rate) and y (Quad Class 'predict_for' real Growth Rate)

    X = event_by_codes_log.iloc[1:4].values
    y = event_by_codes.iloc[4:][str(predict_for)]

    X = X.reshape(len(X), 4, 1)
    feature = append_previous_timestamps(X)

    # expanding last dimension to apply LSTM/RNN because it needs 2D data
    X = np.reshape(feature, (len(feature), -1, 1))

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    x = r'C:\Users\lenovo\PycharmProjects\FYP\lstm'
    regressor = load_model( x + '\\' +  file_names[predict_for] + '.h5')
    y_pred = regressor.predict(X)

    result = y_pred

    predicted_growth_rate = result[0][0]

    df_list = event_by_codes.iloc[1:4].values.tolist()
    n_articles_yesterday = df_list[2][predict_for - 1]
    n_articles_today = 0

    y = y.tolist()
    if len(y) > 0:
        # this means that the day was not tomorrow
        n_articles_today = y[0]
    '''
    growth_rate = log(n_articles_today / n_articles_yesterday)
    n_articles_today = log_inv(growth_rate) * n_articles_yesterday
    '''

    predicted_n_articles = np.exp(predicted_growth_rate) * n_articles_yesterday

    output = {'data': df_list, 'n_articles_yesterday': n_articles_yesterday, 'n_articles_today': n_articles_today, 'predicted_n_articles':predicted_n_articles.round(), 'days': [day1, day2, day3]}
    print(json.dumps(output))


if __name__ == '__main__':
    main()

