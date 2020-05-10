import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# parameters
window = 30
neededCol = ['user/throttle', 'as5048a']
# neededCol = ['user/throttle','user/angle','bno055/heading', 'bno055/roll', 'bno055/pitch', 'ads1115/vm', 'as5048a']

# dataset
train_X = np.zeros((0, window, len(neededCol)), dtype=np.float64)
train_Y = np.zeros((0,), dtype=np.float64)
test_X = np.zeros((0, window, len(neededCol)), dtype=np.float64)
test_Y = np.zeros((0,), dtype=np.float64)

# get all data file path
dataFiles = []
for root, dirs, files in os.walk(r'.\data'):
    for name in files:
        dataFiles.append(os.path.join(root, name))


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    dff = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(dff.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(dff.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


for dataFile in dataFiles:
    # load data from file
    df = pd.read_csv(dataFile)
    df.sort_values('milliseconds', inplace=True)
    df['milliseconds'] = pd.to_datetime(df['milliseconds'], unit='s')
    df.set_index('milliseconds', inplace=True)
    del df['index']

    # reframe
    rule_type = '50ms'
    try:
        resampled = df.resample(rule_type).bfill()
    except ValueError:
        print(dataFile)

    # needed columns
    vc = resampled[neededCol]
    values = vc.values

    # normalize features
    scaler = MinMaxScaler()
    scaler.fit(values)
    scaled = scaler.transform(values)

    # reframe series to train data
    reframed = series_to_supervised(scaled, window, 1)
    values = reframed.values
    # print(values.shape)

    # split the train and test
    # use the middle 1/10 as test set
    share = values.shape[0] // 10
    p1 = share * 5
    p2 = share * 6
    train1 = values[:p1, :]
    test = values[p1:p2, :]
    train2 = values[p2:, :]
    train = np.concatenate((train1, train2))

    # get lable
    train_x, train_y = train[:, :-len(neededCol)], train[:, -1]
    test_x, test_y = test[:, :-len(neededCol)], test[:, -1]
    # reshape NxTxD
    train_x = train_x.reshape((train_x.shape[0], window, len(neededCol)))
    test_x = test_x.reshape((test_x.shape[0], window, len(neededCol)))
    # print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

    train_X = np.concatenate((train_X, train_x))
    train_Y = np.concatenate((train_Y, train_y))
    test_X = np.concatenate((test_X, test_x))
    test_Y = np.concatenate((test_Y, test_y))

print(train_X.shape, train_Y.shape, test_X.shape, test_Y.shape)


from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# make the RNN
i = Input(shape=(train_X.shape[1], train_X.shape[2]))
x = LSTM(100)(i)  # 10 is hidden feature dimensionality
x = Dropout(0.2)(x)
x = Dense(1)(x)  # 1 is output size
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(),
)

# train the RNN
r = model.fit(
  train_X, train_Y,
  batch_size=70,
  epochs=50,
  validation_data=(test_X, test_Y),
)

import matplotlib.pyplot as plt
# plot some data
plt.plot(r.history['loss'], label='loss')
plt.plot(r.history['val_loss'], label='val_loss')
plt.legend()

plt.show()

model_path = 'weight.ckpt'
model.save_weights(model_path)



