import matplotlib.pyplot as plt
import numpy as np
x = [0.1, 0.3, 0.5, 0.7, 0.9]
y1 = [0.402, 0.444, 0.494, 0.494, 0.533]
y2 = [0.877, 0.931, 0.929, 0.918, 0.836]

plt.title('Result of Gaussian kernel')
plt.xlabel('gamma')
plt.ylabel('accuracy')

plt.plot(x, y1, marker='o', markersize=3)
plt.plot(x, y2, marker='o', markersize=3)



plt.legend(['CIFAR-10', 'MNIST'])
plt.show()