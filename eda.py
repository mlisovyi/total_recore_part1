#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import getenv
from os.path import abspath, join

#%%
data_dir = getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
df = pd.read_csv(
    join(data_dir, "public.csv.gz"),
    index_col=0,
)
# %%
df_min_depth = df.groupby("SiteID")["DepthFrom"].min()
df_max_depth = df.groupby("SiteID")["DepthTo"].max()
df_step_depth = df["DepthTo"] - df["DepthFrom"]
# %%
# DEPTH stats
df_min_depth.plot.hist(bins=20)
plt.show()
df_max_depth.plot.hist(bins=20)
plt.show()
df_step_depth.plot.hist(bins=20)
plt.show()
# %%
# Some spectra
n = 10
df_ = df.sort_values(by="CuPPM").tail(n)
for i in range(n):
    v_max = df_.iloc[i, 100:-250].max()
    v_min = df_.iloc[i, 100:-250].min()
    (df_.iloc[i, 3:-49].clip(v_min, v_max) / v_max).plot(label=str(i))
    plt.legend()
plt.title("Top CuPPM")
plt.show()

# %%
n = 5
df_ = df.sort_values(by="CuPPM").head(n)
for i in range(n):
    v_max = df_.iloc[i, 100:-250].max()
    v_min = df_.iloc[i, 100:-250].min()
    (df_.iloc[i, 3:-49].clip(v_min, v_max) / v_max).plot(label=str(i))
    plt.legend()
plt.title("Bottom CuPPM")
plt.show()
# %%
