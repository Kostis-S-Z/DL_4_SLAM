import matplotlib.pyplot as plt
import numpy as np


x = [0.0, 0.1,0.2,0.3,0.4,0.5, 0.6, 0.7,0.8, 0.9]
auc = [0.698, 0.697, 0.722, ]
f1r1 = [0.362, 0.337, 0.344, 0.357, 0.366, 0.363, 0.365, 0.337, 0.316, 0.128]
f1r2 = [0.325, 0.389, 0.360, 0.387, 0.380, 0.362, 0.375, 0.371, 0.352, 0.171]
f1 = []
f1_std = []
for elem1, elem2 in zip(f1r1, f1r2):
    f1.append(np.mean([elem1, elem2]))
    f1_std.append(np.std([elem1, elem2]))
mcc = [0.]
plt.errorbar(x, f1, yerr=f1_std)
plt.xlabel('Dropout percentage')
plt.ylabel('f1 score')
plt.title('Dropout tuning')
plt.show()