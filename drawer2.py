import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

word_list = [
    # "warm_up",
    "walk",
    "run",
    "jump",
    "drink",
    # "lift dumbbell",
    "sit",
    "eat",
    # "turn steering wheel",
    "phone",
    "boxing",
    "throw",
]

m1, m2 = [], []

for w in word_list:
    v1, v2 = np.random.randint(100, size=[233]), np.random.randint(100, size=[233])
    m1.append(v1)
    m2.append(v2)

m1, m2 = np.array(m1), np.array(m2)

print("before PCA:", m1.shape)

pca = PCA(n_components=2)
pca.fit(m1)
m1 = pca.fit_transform(m1)
pca.fit(m2)
m2 = pca.fit_transform(m2)

print("after PCA:", m1.shape)

plt.figure(figsize=(15, 6))

ax = plt.subplot(1, 2, 1)
x = m1[:, 0]
y = m1[:, 1]
ax.scatter(x, y)
ax.set_title("Word Vectors")
for i in range(len(x)):
    ax.text(x[i]+1, y[i]+1, word_list[i]) #给散点加标签

ax = plt.subplot(1, 2, 2)
x = m2[:, 0]
y = m2[:, 1]
ax.scatter(x, y)
ax.set_title("Pose Priors")
for i in range(len(x)):
    ax.text(x[i]+1, y[i]+1, word_list[i]) #给散点加标签

plt.show()
