from keras.models import Model
from keras.layers import Dense, Input, multiply, concatenate, Dropout
from keras.regularizers import l2
import numpy as np
import random
import keras.backend as K
from tensorflow_core.python.keras.backend import _constant_to_tensor
from tensorflow_core.python.ops import clip_ops


def log_loss(y_true, y_pred):
    y_pred = K.constant(y_pred) if not K.is_tensor(y_pred) else y_pred
    y_true = K.cast(y_true, y_pred.dtype)
    epsilon_ = _constant_to_tensor(K.epsilon(), y_pred.dtype.base_dtype)
    y_pred = clip_ops.clip_by_value(y_pred, epsilon_, 1. - epsilon_)
    loss = y_true * K.log(y_pred + K.epsilon())
    loss += (y_true - 1) * K.log(1 - y_pred + K.epsilon())
    return K.mean(abs(loss))


# Load data
train_wic1 = np.array(np.load('../../processed_data/train_wic1.npy'))
train_wic2 = np.array(np.load('../../processed_data/train_wic2.npy'))
dev_wic1 = np.array(np.load('../../processed_data/dev_wic1.npy'))
dev_wic2 = np.array(np.load('../../processed_data/dev_wic2.npy'))
train_labels = np.load('../../processed_data/train_labels.npy')
dev_labels = np.load('../../processed_data/dev_labels.npy')
max_length = 1024
old_train_wic1 = train_wic1
old_train_wic2 = train_wic2
old_train_labels = train_labels

layers = [1024, 512, 256]
num_layer = len(layers)  # Number of layers in the MLP
dropout = 0.5
reg = 0.0001
ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
result = []
for ratio in ratios:
    neg_train_wic1 = []
    neg_train_wic2 = []
    neg_train_labels = []
    train_prim_weights = np.ones(len(old_train_labels))
    neg_train_weights = []
    for i in range(len(old_train_labels)):
        for k in range(ratio):
            neg_train_wic1.append(old_train_wic1[i])
            neg_train_wic2.append(random.choice(old_train_wic2))
            neg_train_labels.append(np.float(0.5))
            neg_train_weights.append(np.float(0.0))
    train_wic1 = np.concatenate((old_train_wic1, np.array(neg_train_wic1)))
    train_wic2 = np.concatenate((old_train_wic2, np.array(neg_train_wic2)))
    train_labels = np.concatenate((old_train_labels, np.array(neg_train_labels)))
    train_sample_weights = np.concatenate((train_prim_weights, np.array(neg_train_weights)))

    # Input variables
    wic1_input = Input(shape=(max_length,), dtype='float32')
    wic2_input = Input(shape=(max_length,), dtype='float32')

    # MF part
    mf_vector = multiply([wic1_input, wic2_input])
    mf_vector = Dropout(dropout)(mf_vector)

    # MLP part
    mlp_vector = concatenate([wic1_input, wic2_input])
    mlp_vector = Dropout(dropout)(mlp_vector)
    for idx in range(0, num_layer):
        layer = Dense(layers[idx], activation='relu', kernel_regularizer=l2(0.001))
        mlp_vector = layer(mlp_vector)
        mlp_vector = Dropout(dropout)(mlp_vector)

    # Concatenate MF and MLP parts
    predict_vector = concatenate([mf_vector, mlp_vector])

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(predict_vector)

    model = Model(input=[wic1_input, wic2_input],
                  output=prediction)
    # Compile the model
    model.compile(optimizer='adam', loss=log_loss, weighted_metrics=['accuracy'])

    # Fit the model
    history = model.fit(
        [train_wic1, train_wic2],
        train_labels, epochs=5, verbose=2,
        shuffle=False,
        sample_weight=train_sample_weights)

    result.append(model.evaluate([dev_wic1, dev_wic2], dev_labels))

np.save('negative_sample/result.npy', result)


