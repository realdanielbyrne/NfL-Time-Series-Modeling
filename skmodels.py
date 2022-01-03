# %%
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.utils.plotting import plot_series
import pandas as pd


# teams conpared
teams = ['car', 'nor']

# historical data range [start,stop]
# end points are inclusive
years = ['2018', '2021']

# forecasting horizon
fh = 1


def read_data():
    teamfile = './stats/combinedstats/{team}_{start}_{stop}.csv'.format(team=teams[0],start=years[0],stop=years[1])
    teamdf = pd.read_csv(teamfile)

    oppfile = './stats/combinedstats/{opp}_{start}_{stop}.csv'.format(opp=teams[1],start=years[1],stop=years[1])
    oppdf = pd.read_csv(oppfile)
    return [teamdf, oppdf]



# step 3: specifying the forecasting algorithm
models = {
    "bats":BATS(
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=True,
        use_arma_errors=True,
        sp=12),

    "tbats":TBATS(
        use_box_cox=False,
        use_trend=True,
        use_damped_trend=True,
        use_arma_errors=True,
        sp=12),

    "aarima":AutoARIMA(sp=12, d=0, max_p=2, max_q=2, suppress_warnings=True),
    "exp":ExponentialSmoothing(trend='add', seasonal='additive', sp=12),
    "poly":PolynomialTrendForecaster(degree=1)
}

#%%
# Team Predictions

def get_teams():
    predictions = pd.DataFrame()
    for t in teams:
        exo = teamdf[['Off_Totyd','Off_TO','Def_TO','PassY/A','PassCmp','RushAtt', 'RushYds', 'RushY/A', 'FGM','FGA','Pnt','PntYds']].copy() 
        target = teamdf['TmScore']
        holdout = teamdf['TmScore']
        exo_mu = pd.DataFrame(exo.mean(axis=0)).transpose()
        
        for m in models:
            models[m].fit(target, X=exo)
            predictions[m]= models[m].predict(fh, X=exo_mu)
            predictions['team'] = t


#%%
# Opp Predictions
oppexo = oppdf[['Off_Totyd','Off_TO','Def_TO','PassY/A','PassCmp','RushAtt', 'RushYds', 'RushY/A', 'FGM','FGA','Pnt','PntYds']].copy() 
target = oppdf['TmScore']
holdout = oppdf['TmScore']
oppexo_mu = pd.DataFrame(oppexo.mean(axis=0)).transpose()

opppredictions = pd.DataFrame()
for m in models:
    models[m].fit(target, X=oppexo)
    opppredictions[m]= models[m].predict(fh, X=oppexo_mu)
    opppredictions['team'] = opp  

opppredictions

#%%
# Spreads





# %%
ou = predictions.to_numpy() + opppredictions.to_numpy()
ou

