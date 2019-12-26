import matplotlib.pyplot as plt
import numpy as np


dropouts = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
acc = np.load('acc.npy')
loss = np.load('loss.npy')

xi = list(range(len(dropouts)))

plt.plot(acc)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Dropout')
plt.xticks(xi, dropouts)
# plt.savefig('dropout_acc.png')
plt.show()

plt.plot(loss)
plt.title('Model loss')
plt.xticks(xi, dropouts)
plt.ylabel('Loss')
plt.xlabel('Dropout')
# plt.savefig('dropout_loss.png')
plt.show()
