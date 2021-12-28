# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pycaret.time_series import *


path = './stats/combinedstats/dal_2018_2021.csv'
df = pd.read_csv(path)


#%%
#pick columns
df1 = df[['TmScore']]
test = df1.tail(1)
df1 = df1[:-1]



# %%
from pycaret.time_series import *
exp_name = setup(data = df1,  fh = 1,seasonal_period='W')

# %%
best_model = compare_models(sort = 'SMAPE')


# %%
et_model = create_model('et_cds_dt')

# %%
knn_model = create_model('knn_cds_dt')

# %%
aa_model = create_model('auto_arima')

# %%
lightgbm_model = create_model('lightgbm_cds_dt')

# %%
tuned_et_model = tune_model(et_model)
tuned_knn_model = tune_model(knn_model)
tuned_aa_model = tune_model(aa_model)
tuned_lightgbm_model = tune_model(lightgbm_model)

# %%
plot_model(tuned_et_model)

# %%
plot_model(tuned_knn_model)

# %%
plot_model(tuned_aa_model)

# %%
plot_model(tuned_lightgbm_model)

# %%
blend_specific = blend_models(estimator_list = [tuned_lightgbm_model,tuned_aa_model,tuned_knn_model,tuned_et_model])

# %%
plot_model(blend_specific)

# %%
preds = predict_model(blend_specific);
preds


# %%
pred_unseen= predict_model(finalize_model(blend_specific));
pred_unseen