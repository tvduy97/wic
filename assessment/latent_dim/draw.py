import matplotlib.pyplot as plt
import numpy as np


acc = np.load('acc.npy')
loss = np.load('loss.npy')

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
