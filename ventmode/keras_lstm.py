import argparse

from keras import optimizers
from keras.layers import Activation, Dense, LSTM
from keras.losses import categorical_crossentropy
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import RobustScaler

from main import FEATURE_SETS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    args = parser.parse_args()

    dataset = pd.read_pickle(args.dataset)
    dataset = dataset[dataset.y.isin([0, 1, 3, 4, 6])]
    # map to new numbers to make simpler for now
    dataset.loc[dataset[dataset.y == 3].index, 'y'] = 2
    dataset.loc[dataset[dataset.y == 4].index, 'y'] = 3
    dataset.loc[dataset[dataset.y == 6].index, 'y'] = 4
    train_set = dataset[dataset.set_type == 'train']
    test_set = dataset[dataset.set_type == 'test']
    x_train = train_set[FEATURE_SETS['vfinal']]
    y_train = train_set.y
    scaler = RobustScaler()
    x_train = scaler.fit_transform(x_train.values)
    # The way machinelearningmastery defines this is [samples, time steps, features]
    # It seems like they interchange the features and time steps slots to do some
    # modeling experimentation. I dunno if that modeling approach makes as much
    # sense for this actual problem tho.
    batch_size = 1
    x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
    y_train = to_categorical(y_train.values)

    model = Sequential()
    # keras asks for a clean batch division
    divisor = len(x_train) / batch_size
    x_train = x_train[:divisor*batch_size]
    y_train = y_train[:divisor*batch_size]
    model.add(LSTM(32, batch_input_shape=(batch_size, 1, 7), stateful=True))
    model.add(Dense(5))
    model.add(Activation('softmax'))
    optim = optimizers.SGD(lr=0.001, decay=1e-6, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=optim)
    for i in range(1):
        model.fit(x_train, y_train, epochs=1, batch_size=batch_size, shuffle=False)
        model.reset_states()

    x_test = scaler.transform(test_set[FEATURE_SETS['vfinal']].values)
    x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
    divisor = len(x_test) / batch_size
    x_test = x_test[:divisor*batch_size]
    y_test = test_set.y.values[:divisor*batch_size]
    preds = model.predict(x_test, batch_size=batch_size)
    print(classification_report(y_test, np.argmax(preds, axis=1)))


if __name__ == "__main__":
    main()
