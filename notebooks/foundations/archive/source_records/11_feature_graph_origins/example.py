import pandas as pd

from feature_graph.features import get_base_features, get_feature_graph
from feature_graph.project_constants import baseline_df_file, df_file, axes
from feature_graph.operators import enter, exit, between, operators

baseline_df = pd.read_csv(baseline_df_file)
df = pd.read_csv(df_file, nrows=500000)

df = df.rename(columns=axes)
df = df[axes.values()]

base_features = get_base_features(axes, df, 'fault_number')
feature_graph = get_feature_graph(base_features, operators)
