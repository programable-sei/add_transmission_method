import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# fig = plt.figure(figsize=(12, 8))

Frequency = np.linspace(60000, 30000000, 501)


df = pd.read_csv("C1XMEM01.CSV", usecols=[0], header=None)
length = 11
 
df = np.log(10 ** (abs(df)/20))/length
# df = -1 / length * np.log( abs(df) )/length
df = df.values
# print(type(df))

df_sort = df 
# df_sort = np.sort(df)[::-1]

# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel("Frequency")
# ax.set_ylabel("Propagation")

# range = (30000000 - 60000)/6

# ax.plot(Frequency, df_sort, color="red", label="propa")

# ax.set_xlim([60000, 30000000])
# ax.set_xticks([60000, 60000 + range, 60000 + 2*range, 60000 + 3*range, 60000 + 4*range, 60000 + 5*range, 30000000])

# fig.tight_layout()              #レイアウトの設定
# plt.show()