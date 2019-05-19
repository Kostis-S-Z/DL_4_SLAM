import matplotlib.pyplot as plt
import numpy as np


x = [0.0, 0.1,0.2,0.3,0.4,0.5, 0.6, 0.7,0.8]
auc = [0.698, 0.697, 0.722, ]
f1r1311 = [0.362, 0.337, 0.344, 0.357, 0.366, 0.363, 0.365, 0.337, 0.316]
f1r1417 = [0.325, 0.389, 0.360, 0.387, 0.380, 0.362, 0.375, 0.371, 0.352]
f1r1517 = [0.324, 0.353, 0.351, 0.396, 0.375, 0.366, 0.353, 0.341, 0.302]
f1r1618 = [0.047, 0.384, 0.362, 0.378, 0.370, 0.353, 0.346, 0.347, 0.293]
f1r1719 = [0.029, 0.373, 0.368, 0.383, 0.363, 0.347, 0.346, 0.329, 0.291]
f1r1820 = [0.138, 0.373, 0.339, 0.391, 0.364, 0.336, 0.352, 0.346, 0.290]
f1 = []
f1_std = []
for elem1, elem2, elem3, elem4, elem5, elem6 in zip(f1r1311, f1r1417, f1r1517, f1r1618, f1r1719, f1r1820):
    f1.append(np.mean([elem1, elem2, elem3, elem4, elem5, elem6]))
    f1_std.append(np.std([elem1, elem2, elem3, elem4, elem5, elem6]))
mcc = [0.]
plt.errorbar(x, f1, color='#0000A0', linewidth=2, yerr=f1_std,  ecolor='#43C6DB', elinewidth=1, capsize=3, barsabove=True, capthick=1)
plt.axhline(0.190, color='gray', linestyle='--',  linewidth = 1, label='Baseline')
plt.xlabel('Dropout percentage', fontsize = 18)
plt.ylabel('F1 score', fontsize=18)
plt.legend()
plt.tick_params(axis='both', which='major', labelsize=16)
plt.tick_params(axis='both', which='minor', labelsize=16)
plt.title('Dropout', fontsize=20)
plt.tight_layout()
plt.show()


