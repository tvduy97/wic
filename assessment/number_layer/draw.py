import matplotlib.pyplot as plt
import numpy as np


acc = np.load('acc.npy')
loss = np.load('loss.npy')

layer_lb = [1, 2, 3, 4, 5, 6, 7]
xi = list(range(len(acc)))

plt.plot(acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number MLP layer')
plt.xticks(xi, layer_lb)
plt.show()

plt.plot(loss)
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Number MLP layer')
plt.xticks(xi, layer_lb)
plt.show()
