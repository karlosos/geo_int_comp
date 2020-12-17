import pandas as pd
import matplotlib.pyplot as plt

df_brama = pd.read_csv("./experiments/exp_block_sizes_brama_0.5.csv")
df_brama = df_brama.drop(['Unnamed: 0'], axis=1)

df_obrotnica = pd.read_csv("./experiments/exp_block_sizes_obrotnica_0.5.csv")
df_obrotnica = df_obrotnica.drop(['Unnamed: 0'], axis=1)

df_wraki = pd.read_csv("./experiments/exp_block_sizes_wraki_utm_0.05_idw.pckl.csv")
df_wraki = df_wraki.drop(['Unnamed: 0'], axis=1)

x = df_brama["block_size"]
plt.subplot(211)
plt.plot(x, df_brama["time"], label="brama")
plt.plot(x, df_obrotnica["time"], label="obrotnica")
plt.plot(x, df_wraki["time"], label="wraki")
plt.ylabel("Czas kompresji [s]")
plt.legend()

plt.subplot(212)
plt.plot(x, df_brama["mean_error"], label="brama")
plt.plot(x, df_obrotnica["mean_error"], label="obrotnica")
plt.plot(x, df_wraki["mean_error"], label="wraki")
plt.ylabel("Średni błąd [m]")
plt.legend()
plt.xlabel("Rozmiar bloku kompresji")

plt.tight_layout()
plt.show()
