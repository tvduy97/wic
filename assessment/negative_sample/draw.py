import matplotlib.pyplot as plt
import numpy as np


ratios = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

results = np.load('result.npy')
acc = []
loss = []

for result in results:
    loss.append(result[0])
    acc.append(result[1])

xi = list(range(len(ratios)))
plt.plot(acc)
plt.xticks(xi, ratios)
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Number of negatives')
plt.show()

plt.plot(loss)
plt.title('Model loss')
plt.xticks(xi, ratios)
plt.ylabel('Loss')
plt.xlabel('Number of negatives')
plt.show()
