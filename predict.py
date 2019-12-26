from keras.models import Model
from keras.layers import Dense, Input, multiply, concatenate, Dropout
from keras.regularizers import l2
import numpy as np

# Load data
test_wic1 = np.array(np.load('processed_data/test_wic1.npy'))
test_wic2 = np.array(np.load('processed_data/test_wic2.npy'))
max_length = 1024

# Build model
layers = [1024, 512, 256, 128]
num_layer = len(layers)  # Number of layers in the MLP
dropout = 0.5
reg = 0.001
# Input variables
wic1_input = Input(shape=(max_length,), dtype='float32')
wic2_input = Input(shape=(max_length,), dtype='float32')

# GMF part
gmf_vector = multiply([wic1_input, wic2_input])
gmf_vector = Dropout(dropout)(gmf_vector)

# MLP part
mlp_vector = concatenate([wic1_input, wic2_input])
mlp_vector = Dropout(dropout)(mlp_vector)
for idx in range(0, num_layer):
    layer = Dense(layers[idx], activation='relu', kernel_regularizer=l2(0.001))
    mlp_vector = layer(mlp_vector)
    mlp_vector = Dropout(dropout)(mlp_vector)

# Concatenate GMF and MLP parts
neumf_vector = concatenate([gmf_vector, mlp_vector])

# Output layer
output = Dense(1, activation='sigmoid', init='lecun_uniform', name="prediction")(neumf_vector)

model = Model(input=[wic1_input, wic2_input],
              output=output)
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Load weights
model.load_weights('save_model_neumf.hdf5')

# Predict
results = model.predict([test_wic1, test_wic2])

# Write to file
labels = []
for p in results:
    if p[0] > 0.5:
        labels.append('T')
    else:
        labels.append('F')
with open('output.txt', 'w') as f:
    for item in labels:
        f.write("%s\n" % item)
