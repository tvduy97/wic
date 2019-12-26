import matplotlib.pyplot as plt
import numpy as np


kernel_regularizers = [
    'None',
    'l2(0.1)', 'l2(0.01)', 'l2(0.001)', 'l2(0.0001)', 'l2(0)',
    'l1(0.1)', 'l1(0.01)', 'l1(0.001)', 'l1(0.0001)', 'l1(0)'
]

acc = np.load('acc.npy')
loss = np.load('loss.npy')

xi = list(range(len(kernel_regularizers)))
plt.plot(acc)
plt.xticks(xi, kernel_regularizers, rotation=50)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Kernel regularizer')
plt.show()

plt.plot(loss)
plt.title('Model loss')
plt.xticks(xi, kernel_regularizers, rotation=50)
plt.ylabel('Loss')
plt.xlabel('Kernel regularizer')
plt.show()
