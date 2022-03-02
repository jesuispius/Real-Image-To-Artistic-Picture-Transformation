import numpy as np

A = np.pad(np.array([1, 2, 3]), (3 // 2, 3 // 2), 'constant')
print(A)
