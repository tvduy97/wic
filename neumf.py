from keras.models import Model
from keras.layers import Dense, Input, multiply, concatenate, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np


# Load data
train_wic1 = np.array(np.load('processed_data/train_wic1.npy'))
train_wic2 = np.array(np.load('processed_data/train_wic2.npy'))
dev_wic1 = np.array(np.load('processed_data/dev_wic1.npy'))
dev_wic2 = np.array(np.load('processed_data/dev_wic2.npy'))
train_labels = np.load('processed_data/train_labels.npy')
dev_labels = np.load('processed_data/dev_labels.npy')

# Build model
layers = [1024, 512, 256]
num_layer = len(layers)  # Number of layers in the MLP
dropout = 0.5
reg = l2(0.0001)
max_length = 1024
# Input variables
wic1_input = Input(shape=(max_length,), dtype='float32')
wic2_input = Input(shape=(max_length,), dtype='float32')

wic1_latent = wic1_input
wic2_latent = wic2_input
# wic1_latent = Dense(128, activation='relu')(wic1_input)
# wic2_latent = Dense(128, activation='relu')(wic2_input)

# GMF part
gmf_vector = multiply([wic1_latent, wic2_latent])
gmf_vector = Dropout(dropout)(gmf_vector)

# MLP part
mlp_vector = concatenate([wic1_latent, wic2_latent])
mlp_vector = Dropout(dropout)(mlp_vector)
for idx in range(0, num_layer):
    layer = Dense(layers[idx], activation='relu', kernel_regularizer=reg)
    mlp_vector = layer(mlp_vector)
    mlp_vector = Dropout(dropout)(mlp_vector)

# Concatenate MF and MLP parts
neumf_vector = concatenate([gmf_vector, mlp_vector])

# Output layer
output = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(neumf_vector)

model = Model(input=[wic1_input, wic2_input],
              output=output)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# print(model.summary())

# Fit the model
history = model.fit(
    [train_wic1, train_wic2],
    train_labels, epochs=10, verbose=2,
    validation_data=[[dev_wic1, dev_wic2], dev_labels])

# Save weights of model
model.save_weights('save_model_neumf.hdf5')

# Summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
