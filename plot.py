import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

test_idx = [
            16836
        ]
for i, idx in enumerate(test_idx):
    data = np.load(f'v_data_{test_idx}.npy')

    plt.plot(data)
    plt.xlabel('time step')
    plt.ylabel('voltage magnitude (per unit)')
    plt.show()
    print(f'{idx} finish')