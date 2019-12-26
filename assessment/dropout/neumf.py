from keras.models import Model
from keras.layers import Dense, Input, multiply, concatenate, Dropout
import numpy as np


# Load data
train_wic1 = np.array(np.load('../../processed_data/train_wic1.npy'))
train_wic2 = np.array(np.load('../../processed_data/train_wic2.npy'))
dev_wic1 = np.array(np.load('../../processed_data/dev_wic1.npy'))
dev_wic2 = np.array(np.load('../../processed_data/dev_wic2.npy'))
train_labels = np.load('../../processed_data/train_labels.npy')
dev_labels = np.load('../../processed_data/dev_labels.npy')
test_wic1 = np.array(np.load('../../processed_data/test_wic1.npy'))
test_wic2 = np.array(np.load('../../processed_data/test_wic2.npy'))

train_wic1 = np.concatenate((train_wic1, np.array(dev_wic1)))
train_wic2 = np.concatenate((train_wic2, np.array(dev_wic2)))
train_labels = np.concatenate((train_labels, np.array(dev_labels)))

# Build model
layers = [1024, 512]
num_layer = len(layers)  # Number of layers in the MLP
dropout = 0.5
max_length = 1024


# Input variables
wic1_input = Input(shape=(max_length,), dtype='float32')
wic2_input = Input(shape=(max_length,), dtype='float32')

wic1_latent = wic1_input
wic2_latent = wic2_input

# GMF part
gmf_vector = multiply([wic1_latent, wic2_latent])
gmf_vector = Dropout(dropout)(gmf_vector)

# MLP part
mlp_vector = concatenate([wic1_latent, wic2_latent])
mlp_vector = Dropout(dropout)(mlp_vector)
for idx in range(0, num_layer):
    layer = Dense(layers[idx], activation='relu')
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
    train_labels, epochs=10, verbose=2)

model.save_weights('save_model_neumf.hdf5')
results = model.predict([test_wic1, test_wic2])
labels = []
for p in results:
    if p[0] > 0.5:
        labels.append('T')
    else:
        labels.append('F')
with open('output.txt', 'w') as f:
    for item in labels:
        f.write("%s\n" % item)
