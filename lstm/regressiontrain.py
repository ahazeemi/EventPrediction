from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from sklearn import  metrics
import os
from keras.models import load_model
from keras.layers import Input, Dense,Activation, LSTM, MaxPooling1D, Flatten, LeakyReLU, Dropout,ConvLSTM2D
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping


# Let's use time component and transform data with log
def log_scale_dataframe(df):
    new_df = pd.DataFrame()
    for col in df.columns:
        new_df[col] = np.log(df[col] / df[col].shift() )
    return new_df


# Prepare dataset  - take events for country EG
def split_events(gdelt_df, country_code, measure_name = 'NumArticles'):
    event_root_codes = gdelt_df.QuadClass.unique()

    event_series = [
        gdelt_df[(gdelt_df.QuadClass == event_code) & (gdelt_df.Actor1Geo_CountryCode == country_code)][[measure_name]]
        for event_code in event_root_codes]
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

    output = Dense(1)(x)  # 5 output units because 5 possible predictions are there

    model = Model(input_, output)

    model.compile(optimizer='adam', loss='mean_squared_error')  # compile the model
    return model


def append_previous_timestamps(X, labels):

    # This function appends previous three timestamps
    label = []
    feature = []
    j = 1
    k = 2

    for i in range(7026):
        features = [X[k][0], X[k][1], X[k][2], X[k][3], X[j][0], X[j][1], X[j][2], X[j][3], X[i][0], X[i][1], X[i][2],
                    X[i][3]]
        feature.append(features)
        label.append(labels[k])
        j = j + 1
        k = k + 1
    feature = np.array(feature)
    label = np.array(label)

    return feature, label


def main():

    # PARAMETERS
    input_file = 'fullevents.csv'
    plot_rawdata = False
    country_code = 'PK'
    train = False
    test = True

    '''
    1: Material Cooperation
    2: Verbal Cooperation
    3: Verbal Conflict
    4: Material Conflict
    '''
    predict_for = 1   # predicting for material conflict

    pd.options.display.float_format = '{:.2f}'.format
    plt.style.use('ggplot')

    gdelt_df = pd.read_csv(input_file)
    print (gdelt_df.info())
    gdelt_df.describe()

    gdelt_df['SQLDATE']=gdelt_df['SQLDATE'].apply(str)
    gdelt_df['Date'] = pd.to_datetime(gdelt_df['SQLDATE'])
    gdelt_df.index = gdelt_df['Date']
    gdelt_df.drop(['SQLDATE'], axis=1, inplace=True)

    event_by_codes = split_events(gdelt_df, country_code)

    if plot_rawdata:
        plot_raw_data(event_by_codes)
        exit()

    event_by_codes_log = log_scale_dataframe(event_by_codes)

    # Prepare X (all quads classes' Growth Rate) and y (Quad Class 'predict_for' real Growth Rate)
    X = event_by_codes_log.values
    y = event_by_codes_log[str(predict_for)].shift(periods=-1).values

    # There no info from the past for the very first day.
    # Let's drop out first day.
    X = X[1:-1]
    y = y[1:-1]

    X = X.reshape(len(X), 4, 1)
    print("X shape: {}, y shape: {}".format(X.shape, y.shape))

    feature, label = append_previous_timestamps(X, y)

    # expanding last dimension to apply LSTM/RNN because it needs 2D data
    feats = np.reshape(feature, (len(feature), -1, 1))
    feats1 = np.reshape(X, (len(X), -1, 1))
    # labels=np.reshape(labels,(len(labels),-1,1))
    # splitting data with 0.2% size of cross validation data
    X_train, X_test, y_train, y_test = train_test_split(feats, label, test_size=0.2, random_state=42, shuffle=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if train:
        model = prepare_lstm()
        es = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto',
                                           baseline=None, restore_best_weights=True)
        # train the model
        model.fit(X_train, y_train,
                  epochs=50,
                  batch_size=200,
                  shuffle=False,
                  validation_data=(X_test, y_test)
                  , callbacks=[es])
        model.save('models/' + country_code + str(predict_for) + 'lstmregression.h5')  # save the model

    if test:
        regressor = load_model('models/' + country_code + str(predict_for) + 'lstmregression.h5')
        # Predict Growth Rate values on Test part
        y_pred = regressor.predict(X_test)
        # Plot prediction results and MSE between real Growth Rate of Test part and predicted values
        plt.plot(y_test, label='True values')
        plt.plot(y_pred, label='Predicted values')
        plt.legend()
        plt.show()
        print("MSE: {}".format(metrics.mean_squared_error(y_test, y_pred)))


if __name__ == '__main__':
    main()

