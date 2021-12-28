# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
import statsmodels.api as sm
import numpy as np 
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import imageio
import os
from statsmodels.graphics.tsaplots import plot_acf

path = './stats/combinedstats/dal_2018_2021.csv'
df1 = pd.read_csv(path)

path = './stats/combinedstats/was_2018_2021.csv'
df2 = pd.read_csv(path)


# %%
df1.head()

# %%
df2.head()

# %%
import plotly.express as px
fig = px.line(df1, x=df1.index, y=["TmScore", "Avg3Game","Avg6Game" ], template = 'plotly_dark')
fig.show()

# %%
import plotly.express as px
fig = px.line(df2, x=df2.index, y=["TmScore", "Avg3Game","Avg6Game" ], template = 'plotly_dark')
fig.show()

#%%
#pick columns
team1 = df1[['Off_Totyd','Off_TO','Def_TO','PassY/A','PassCmp','RushAtt', 'RushYds', 'RushY/A','month','year','@','TmScore','Opp']]
team2 = df2[['Off_Totyd','Off_TO','Def_TO','PassY/A','PassCmp','RushAtt', 'RushYds', 'RushY/A','month','year','@','TmScore','Opp']]

# %%
train1 = team1[:-1]
train2 = team2[:-1]
test1 = team1[-1:]
test2 = team2[-1:]


# %%
# import the regression module
from pycaret.regression import *
# initialize setup
s = setup(data = train1, 
            test_data = test1, 
            target = 'TmScore', 
            fold_strategy = 'timeseries', 
            categorical_features = ['month','year','@','Opp'], 
            numeric_features = ['Off_Totyd','Off_TO','Def_TO','PassY/A','PassCmp','RushAtt', 'RushYds', 'RushY/A'], 
            combine_rare_levels = True, 
            rare_level_threshold = 0.1,
            remove_multicollinearity = True, 
            multicollinearity_threshold = 0.95,
            fold = 3, 
            session_id = 123,
            pca=False,
            feature_selection=True,
            #bin_numeric_features=True,
            silent = True)

# %%
# returns best models - takes a little time to run
models = compare_models(sort = 'MAE')

# %%
lightgbm = create_model('lightgbm')

# %%
bayesian_ridge = create_model('br')

# %%
tuned_lightgbm = tune_model(lightgbm) 

# %%
tuned_br = tune_model(bayesian_ridge) 

# %%
plot_model(tuned_lightgbm)

# %%
plot_model(tuned_br)

# %%
plot_model(tuned_lightgbm, plot='feature') 

# %%
plot_model(tuned_br, plot='feature') 

# %%
blend_specific = blend_models(estimator_list = [tuned_br,tuned_lightgbm])
# %%
