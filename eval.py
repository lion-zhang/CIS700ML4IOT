import os

from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# parameters
window = 30
neededCol = ['user/throttle', 'as5048a']
# neededCol = ['user/throttle','user/angle','bno055/heading', 'bno055/roll', 'bno055/pitch', 'ads1115/vm', 'as5048a']


# make the RNN
i = Input(shape=(window, len(neededCol)))
x = LSTM(100)(i)  # 10 is hidden feature dimensionality
x = Dropout(0.2)(x)
x = Dense(1)(x)  # 1 is output size
model = Model(i, x)
model.compile(
  loss='mse',
  optimizer=Adam(),
)


model_path = 'weight.ckpt'
model.load_weights(model_path)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# load data from file
df = pd.read_csv('./data/data_1_20200502105458.csv')
df['milliseconds']=pd.to_datetime(df['milliseconds'], unit='s')
df.set_index('milliseconds', inplace=True)
del df['index']

rule_type = '50ms'
resampled = df.resample(rule_type).bfill()
resampled.head()

vc = resampled[neededCol]
values = vc.values

from sklearn.preprocessing import MinMaxScaler
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

mse_lst = []

for p in range(1, 20):
    mtest = scaled[250:750]
    step = p
    predict = []
    target = mtest[window+step:window+step+400, 1]

    for i in range(400):
        input = mtest[i:i+window, :].copy()
        re = model.predict(input.reshape(1, window, len(neededCol)))
        for j in range(step-1):
            input = np.roll(input, -1, axis=0)
            input[-1, 0:-1] = mtest[i+window+j, 0:-1].copy()
            input[-1, -1] = re[0][0]
            re = model.predict(input.reshape(1, window, len(neededCol)))
        predict.append(re[0][0])


    mse = tf.keras.losses.MSE(
        target, predict
    )

    mse_lst.append(mse.numpy())
    plt.plot(predict)
    filename = os.path.join('figure', 'multi', str(p)+'.png')
    plt.savefig(filename)

print(mse_lst)
# 1   0.0036128177

