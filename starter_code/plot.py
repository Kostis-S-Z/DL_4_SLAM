import matplotlib.pyplot as plt
import numpy as np


# x = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.03, 0.1]
# f1r1 = [0.165, 0.199, 0.297, 0.358, 0.382, 0.373, 0.323, 0.221, 0.227]
# f1r2 = [x-0.03 for x in f1r1]
# f1 = []
# f1_std = []
# for elem1, elem2 in zip(f1r1, f1r2):
#     f1.append(np.mean([elem1, elem2]))
#     f1_std.append(np.std([elem1, elem2]))
# plt.xscale("log", nonposx='clip')
# plt.errorbar(x, f1, color='#0000A0', linewidth=2, yerr=f1_std,  ecolor='#43C6DB', elinewidth=1, capsize=3, barsabove=True, capthick=1)
# plt.axhline(0.190, color='gray', linestyle='--',  linewidth = 1, label='Baseline')
# plt.xlabel('Base learning rate for adam')
# plt.ylabel('F1 score')
# plt.legend()
# plt.title('Learning rate')
# plt.show()

import matplotlib.pyplot as plt
import numpy as np


no_emb = [0.308, 0.310,0.311, 0.316]
emb = [0.317,0.316, 0.318, 0.316]
f1 = []
f1_std = []
f1 = [np.mean(no_emb), np.mean(emb)]
f1_std = [np.std(no_emb), np.std(emb)]

N=2
ind = np.arange(N)  # the x locations for the groups
width = 0.35  # the width of the bars: can also be len(x) sequence
plt.bar(ind, f1, width, yerr=f1_std, color='#0000A0', linewidth=2,ecolor='#43C6DB', capsize=3)
plt.axhline(0.190, color='gray', linestyle='--',  linewidth = 1, label='Baseline')
plt.xticks(ind, ('Binary encoding', 'Word embeddings'))
plt.ylabel('F1 score')
#plt.legend()
plt.ylim(0.30, 0.32)
plt.title('Word embeddings')
plt.show()
