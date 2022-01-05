# %%
# imports
from sktime.forecasting.bats import BATS
from sktime.forecasting.tbats import TBATS
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import PolynomialTrendForecaster
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
def get_predictions(teamdata, models, fh):
    
    cols = list(models.keys())
    cols.append('team')
    predictions = pd.DataFrame(columns=cols)
    for d in teamdata:
        exo = d[['Off_Totyd','Off_TO','Def_TO','PassY/A', 'PassCmp', 'RushAtt',
                    'RushYds', 'RushY/A', 'FGM', 'FGA', 'Pnt', 'PntYds']].copy() 
        target = d['TmScore']
        exo_mu = pd.DataFrame(exo.mean(axis=0)).transpose()
        
        prediction = pd.Series(dtype='float64')
        for key, model in models.items():
            model.fit(target, X=exo)
            result = model.predict(fh, X = exo_mu)
            prediction[key] = result.iloc[0]
            
        prediction['team'] = d.loc[0,'team']
        predictions.append(prediction,ignore_index=True, sort=False )
        predictions = predictions.append(prediction, ignore_index=True, sort=False )
    predictions.set_index('team', inplace=True)
        
    return predictions
       
def get_matchups():
    pass 

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
    
    args = parser.parse_args("")

    # teams
    teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
         'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
         'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
         'sea', 'tam', 'oti', 'was']

    # historical data range [start,stop]
    # end points are inclusive

    # step 3: specifying the forecasting algorithm
    models = {

        "tbats":TBATS(
            use_box_cox=False,
            use_trend=True,
            use_damped_trend=False,
            use_arma_errors=True),

        "aarima":AutoARIMA(sp=12, d=0, max_p=4, max_q=2, suppress_warnings=True),
        "exp":ExponentialSmoothing(trend='add', seasonal='additive', sp=12),
        "poly":PolynomialTrendForecaster(degree=3)
    }

    teamdata = read_data(teams, [args.start,args.end])
    preds = get_predictions(teamdata, models, args.fh)
    
# %%
