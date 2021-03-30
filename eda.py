#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import getenv
from os.path import abspath, join
from predict import preprocess

#%%
data_dir = getenv("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
data_file = join(data_dir, "public.csv.gz")
df_pp = preprocess(data_file)
df = pd.read_csv(data_file, index_col=0,)
y_oof = pd.read_csv(join(data_dir, "oof.csv"), index_col=0,)
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
df_ = df_pp.sort_values(by="CuPPM").tail(n)
for i in range(n):
    df_.iloc[i, 3:-49].plot(label=str(i))
    plt.legend()
plt.title("Top CuPPM")
plt.show()

# %%
n = 10
df_ = df_pp.sort_values(by="CuPPM").head(n)
for i in range(n):
    df_.iloc[i, 3:-49].plot(label=str(i))
    plt.legend()
plt.title("Bottom CuPPM")
plt.show()
# %%
