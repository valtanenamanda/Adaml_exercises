import pandas as pd
from scipy.io import arff
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

data, meta = arff.loadarff('mnist_784.arff')
df = pd.DataFrame(data)

X = df.drop(columns=['class']).astype(int)  # pixel values
y = df['class'].astype(int)

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Initialize and train SOM (20x20 map)
som = MiniSom(x=20, y=20, input_len=784, sigma=3, learning_rate=0.5)
som.random_weights_init(X_scaled)
som.train_random(X_scaled, 1000)

fig, ax = plt.subplots(figsize=(8, 8))

# U-matrix 
um_im = ax.pcolor(som.distance_map().T, cmap='bone_r')

# Colorbar 
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="4%", pad=0.25)
cbar = plt.colorbar(um_im, cax=cax, label='Distance')

# Winners
winners = np.array([som.winner(x) for x in X_scaled])

palette = cm.tab10(np.linspace(0, 1, 10))

# Winners plotted as dots in the middle of the squares
for d in range(10):
    mask = (y.values == d)
    ax.scatter(
        winners[mask, 0] + 0.5,
        winners[mask, 1] + 0.5,
        s=8, alpha=0.6, edgecolors='none',
        c=[palette[d]], label=str(d)
    )

ax.set_xlim(0, som._weights.shape[0])
ax.set_ylim(0, som._weights.shape[1])
ax.invert_yaxis()
ax.set_title('SOM U-Matrix with Overlaid Digits (MNIST-784)')

ax.legend(title='Digit', loc='center right', bbox_to_anchor=(-0.12, 0.5),
          frameon=True, borderaxespad=0.0)

plt.tight_layout()
plt.show()
