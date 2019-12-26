from keras.models import Model
from keras.layers import Dense, Input, multiply, concatenate, Dropout
from keras.regularizers import l2
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K


# Load data
train_wic1 = np.array(np.load('../../processed_data/train_wic1.npy'))
train_wic2 = np.array(np.load('../../processed_data/train_wic2.npy'))
dev_wic1 = np.array(np.load('../../processed_data/dev_wic1.npy'))
dev_wic2 = np.array(np.load('../../processed_data/dev_wic2.npy'))
train_labels = np.load('../../processed_data/train_labels.npy')
dev_labels = np.load('../../processed_data/dev_labels.npy')

# Build model
layer_mlp = [1024, 512, 256]
num_layer = len(layer_mlp)
dropout = 0.5
reg = 0.001
max_length = 1024
acc = []
loss = []
latent_dims = [512, 256, 128, 64, 32]

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
    layer = Dense(layer_mlp[idx], activation='relu', kernel_regularizer=l2(reg))
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

los, ac = model.evaluate([dev_wic1, dev_wic2], dev_labels)
acc.append(ac)
loss.append(los)

for latent_dim in latent_dims:
    # Input variables
    wic1_input = Input(shape=(max_length,), dtype='float32')
    wic2_input = Input(shape=(max_length,), dtype='float32')

    # wic1_latent = wic1_input
    # wic2_latent = wic2_input
    wic1_latent = Dense(latent_dim, activation='relu')(wic1_input)
    wic2_latent = Dense(latent_dim, activation='relu')(wic2_input)

    # GMF part
    gmf_vector = multiply([wic1_latent, wic2_latent])
    gmf_vector = Dropout(dropout)(gmf_vector)

    # MLP part
    mlp_vector = concatenate([wic1_latent, wic2_latent])
    mlp_vector = Dropout(dropout)(mlp_vector)
    for idx in range(0, num_layer):
        layer = Dense(layer_mlp[idx], activation='relu', kernel_regularizer=l2(reg))
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

    los, ac = model.evaluate([dev_wic1, dev_wic2], dev_labels)

    acc.append(ac)
    loss.append(los)

np.save('acc.npy', acc)
np.save('loss.npy', loss)

latent_dims = [1024, 512, 256, 128, 64, 32]
xi = list(range(len(latent_dims)))
plt.plot(acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Latent dim')
plt.xticks(xi, latent_dims)
plt.show()

plt.plot(loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Latent dim')
plt.xticks(xi, latent_dims)
plt.show()
