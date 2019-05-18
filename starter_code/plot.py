import matplotlib.pyplot as plt
import numpy as np


x = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]
f1r1 = [0.165, 0.199, 0.297, 0.358, 0.382, 0.373, 0.323, 0.221, 0.227]
f1r2 = [x-0.03 for x in f1r1]
f1 = []
f1_std = []
for elem1, elem2 in zip(f1r1, f1r2):
    f1.append(np.mean([elem1, elem2]))
    f1_std.append(np.std([elem1, elem2]))
plt.xscale("log", nonposx='clip')
plt.errorbar(x, f1, color='#0000A0', linewidth=2, yerr=f1_std,  ecolor='#43C6DB', elinewidth=1, capsize=3, barsabove=True, capthick=1)
plt.axhline(0.190, color='gray', linestyle='--',  linewidth = 1, label='Baseline')
plt.xlabel('Base learning rate for adam')
plt.ylabel('F1 score')
plt.legend()
plt.title('Learning rate')
plt.show()