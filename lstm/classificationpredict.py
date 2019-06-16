from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from keras.models import load_model
from keras.layers import Input, Dense,Activation, LSTM, MaxPooling1D, Flatten, LeakyReLU, Dropout,ConvLSTM2D
from keras.models import Model
from datetime import datetime, timedelta
from keras.utils.vis_utils import plot_model

def value_to_class(x):
    if x <= 0:
        return 0.0
    else:
        return 1.0


value_to_class = np.vectorize(value_to_class)


# Let's use time component and transform data with log
def log_scale_dataframe(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = np.log(df[col] / df[col].shift() )
    return new_df


# Prepare dataset  - take events for country
def split_events(gdelt_df, country_code, day0,  day1, day2, day3):
    event_root_codes = gdelt_df.QuadClass.unique()
    measure_name = 'NumArticles'
    event_series = [gdelt_df[(gdelt_df.QuadClass == event_code) & (gdelt_df.Actor1Geo_CountryCode == country_code)  & ((gdelt_df.Date == day0) | (gdelt_df.Date == day1) | (gdelt_df.Date == day2) | (gdelt_df.Date == day3) )][[measure_name]] for event_code in event_root_codes]

    event_by_codes = pd.concat(event_series, axis=1).sort_index()
    event_by_codes.columns = map(str, event_root_codes)
    event_by_codes = event_by_codes.fillna(method='ffill')
    event_by_codes = event_by_codes.fillna(method='bfill')
    return event_by_codes


def plot_raw_data(data):
    data_plt = data.resample(rule='1M').mean().plot(figsize=(15, 5), fontsize=20)
    plt.show()


def prepare_lstm():
    input_ = Input(shape=(12, 1))  # input layer

    x = LSTM(16, return_sequences=True)(input_)  # lstm layer connected to input
    # x=MaxPooling1D()(x)
    x = Dropout(0.4)(x)

    x = LSTM(32, return_sequences=True)(x)
    # x=MaxPooling1D()(x)
    x = Dropout(0.4)(x)

    x = LSTM(64, return_sequences=True)(input_)  # lstm layer connected to input
    # x=MaxPooling1D()(x)
    x = Dropout(0.4)(x)

    x = LSTM(128, return_sequences=True)(x)
    # x=MaxPooling1D()(x)
    x = Dropout(0.4)(x)

    x = LSTM(256, return_sequences=True)(x)
    # x=MaxPooling1D()(x)

    x = Flatten()(x)  # flattening the output to apply fully connected layers/ dense layers

    x = Dense(256, name='dense3')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(512, name='dense23')(x)
    x = LeakyReLU()(x)
    x = Dropout(0.4)(x)

    x = Dense(1024, name='dense43')(x)
    # x=LeakyReLU()(x)

    output = Dense(2, activation='softmax')(x)  # 5 output units because 5 possible predictions are there

    model = Model(input_, output)

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['acc'])  # compile the model
    return model


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
    input_file = 'fullevents.csv'
    plot_rawdata = False
    country_code = 'PK'
    day3 = '2019-02-28'
    predict_for = 4  # predicting for material conflict

    '''
    1: Material Cooperation
    2: Verbal Cooperation
    3: Verbal Conflict
    4: Material Conflict
    '''

    date = datetime.strptime(day3, "%Y-%m-%d")

    day3 = date.strftime("%Y-%m-%d")
    day2 = (date - timedelta(days=1)).strftime("%Y-%m-%d")
    day1 = (date - timedelta(days=2)).strftime("%Y-%m-%d")
    day0 = (date - timedelta(days=3)).strftime("%Y-%m-%d")

    pd.options.display.float_format = '{:.2f}'.format
    plt.style.use('ggplot')

    gdelt_df = pd.read_csv(input_file)
    # print (gdelt_df.info())
    # gdelt_df.describe()

    gdelt_df['SQLDATE']=gdelt_df['SQLDATE'].apply(str)
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'])
    gdelt_df.index = gdelt_df['Date']
    gdelt_df.drop(['SQLDATE'], axis=1, inplace=True)

    event_by_codes = split_events(gdelt_df, country_code, day0, day1, day2, day3)

    if plot_rawdata:
        plot_raw_data(event_by_codes)

    event_by_codes_log = log_scale_dataframe(event_by_codes)

    # Prepare X (all quads classes' Growth Rate) and y (Quad Class 'predict_for' real Growth Rate)
    X = event_by_codes_log.iloc[1:].values
    y = event_by_codes_log.iloc[3:][str(predict_for)]
    y = value_to_class(y)

    X = X.reshape(len(X), 4, 1)
    print("X shape: {}, y shape: {}".format(X.shape, y.shape))

    feature = append_previous_timestamps(X)

    # expanding last dimension to apply LSTM/RNN because it needs 2D data
    X = np.reshape(feature, (len(feature), -1, 1))

    # splitting data with 0.2% size of cross validation data
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    regressor = load_model('models/' + country_code + str(predict_for) + 'lstmclassification.h5')

    # plot_model(regressor, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    # exit()
    # Predict Growth Rate values on Test part
    y_pred = regressor.predict(X)

    result = y_pred.round()[0][0]

    print(event_by_codes)
    print("Actual: ", y)
    print("Result: ", result)


if __name__ == '__main__':
    main()

