import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
np.random.seed(0)
normal_data_for_seaborn = np.random.normal(loc=5, scale=2, size=200)
df_tips = sns.load_dataset('tips') # Seaborn comes with some sample datasets