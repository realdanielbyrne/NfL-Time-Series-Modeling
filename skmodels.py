# %%
# imports
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
from sktime.forecasting.trend import TrendForecaster
from sktime.forecasting.ets import AutoETS
from sktime.forecasting.compose import EnsembleForecaster
from sktime.utils.plotting import plot_series
import pandas as pd
import argparse

#%%
def read_data(teams, years):
    data = []
    for team in teams:
        teamfile = './stats/combinedstats/{team}_{start}_{stop}.csv'.format(team=team,start=years[0],stop=years[1])
        df = pd.read_csv(teamfile)
        df['team']= team
        data.append(df)

    return data

#%%
def get_forecasts(teamdata, forecasters, fh):

    predictions = pd.DataFrame()
    models = []
    for d in teamdata:
        X = d[['Off_Totyd','Off_TO','Def_TO','PassY/A', 'PassCmp', 'RushAtt',
                    'RushYds', 'RushY/A', 'FGM', 'FGA', 'Pnt', 'PntYds']].copy() 
        y = d['TmScore']
        X_mu = pd.DataFrame(X.mean(axis=0)).transpose()
        
        y_pred = pd.Series(dtype='float64')
        for f in forecasters:
            f[1].fit(y, X=X)
            result = f[1].predict(fh, X = X_mu)
            y_pred[f[0]] = result.iloc[0]
            
        y_pred['team'] = d['team'][0]
        predictions.append(y_pred,ignore_index=True, sort=False )
        predictions = predictions.append(y_pred, ignore_index=True, sort=False )
        models.append(f)
    predictions.set_index('team', inplace=True)
    predictions['mean'] = predictions.mean(axis=1)

    return predictions, models

 # %%      
def get_matchups():
    pass
 
# %%
def get_ensemble_forecasts(teamdata,forecasters,fh):
    forecasts = pd.DataFrame(columns=['team','pred'])
    for d in teamdata:
        X = d[['Off_Totyd','Off_TO','Def_TO','PassY/A', 'PassCmp', 'RushAtt',
                    'RushYds', 'RushY/A', 'FGM', 'FGA', 'Pnt', 'PntYds']].copy() 
        y = d['TmScore']
        X_mu = pd.DataFrame(X.mean(axis=0)).transpose()
        forecaster = EnsembleForecaster(forecasters=forecasters, n_jobs=-1)
        forecaster.fit(y=y, fh=fh, X=X)
        y_pred = forecaster.predict(X=X_mu)
        pred = {'team': d.loc[0,'team'], 'pred':y_pred.iloc[0]}
        forecasts = forecasts.append(pred, ignore_index=True)

    return forecasts 
                

# %%
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='NFL spread Predictor.')
    parser.add_argument('-b','--start',
                        action='store',
                        type=int,
                        default=2018,
                        required=False,
                        help="Historical data range start year.")

    parser.add_argument('-e','--end',
                        type=int,
                        default=2021,
                        required=False,
                        help="Historical data range end year.")
    
    parser.add_argument('-f','--fh',
                        type=int,
                        default=1,
                        required=False,
                        help="Forecast horizon.")
    
    parser.add_argument('-m','--models',
                        type=int,
                        default= [
                            ("tbats",TBATS(
                                use_box_cox=False,
                                use_trend=True,
                                use_damped_trend=False,
                                use_arma_errors=True)),

                            ("aarima",AutoARIMA(sp=12, d=0, max_p=4, max_q=2, suppress_warnings=True)),
                            ("exp",ExponentialSmoothing(trend='add', seasonal='additive', sp=12)),
                            ("poly",PolynomialTrendForecaster(degree=3)),
                            ("trend",TrendForecaster()),
                            ("autoETS",AutoETS())
                        ],
                        required=False,
                        help="Forecast horizon.")
    
    
    args = parser.parse_args("")

    # teams
    teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
        'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
        'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
        'sea', 'tam', 'oti', 'was','nyg']

    forecasters = args.models
    start = args.start
    end = args.end
    fh = args.fh    
    teamdata = read_data(teams, [start,end])
    
    
# %%

    forecasts, models = get_forecasts(teamdata, forecasters, fh)
    forecasts


# %%
    ensembles = get_ensemble_forecasts(teamdata, forecasters, fh)
    ensembles

# %%
