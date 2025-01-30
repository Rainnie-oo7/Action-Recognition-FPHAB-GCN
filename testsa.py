import numpy as np
arr = np.random.rand(45, 21, 3)
corr = []

labels = []
for labelouter in range(45):
    for i in arr:
        corr.append(i)
        labels.append(labelouter)

print()