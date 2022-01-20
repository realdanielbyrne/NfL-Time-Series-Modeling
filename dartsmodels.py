# %%
# imports
import requests
import pandas as pd
import numpy as np
import warnings
import argparse
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.models import (
    ExponentialSmoothing,
    AutoARIMA,
    RegressionModel,
    RandomForest,
    FFT,
    NBEATSModel,
    TFTModel,
    RNNModel,
    TCNModel,
    TransformerModel,
    BlockRNNModel
)

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.filterwarnings("ignore")

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
        X_mu = X.mean(axis=0)
        X = X.append(X_mu, ignore_index=True)
        
        y_ts=TimeSeries.from_series(y)
        X_ts=TimeSeries.from_dataframe(X)
        y_pred = pd.Series(dtype='float64')
    
        for f in forecasters:
            f[1].fit(y_ts, future_covariates=X_ts)
            result = f[1].predict(1, future_covariates=X_ts)
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
                        type=int,
                        default=2018,
                        required=False,
                        help="Historical data range start year.")

    parser.add_argument('-e','--end',
                        type=int,
                        default=2021,
                        required=False,
                        help="Historical data range end year.")
    
    parser.add_argument('-f','--horizon',
                        type=int,
                        default=1,
                        required=False,
                        help="Forecast horizon.")
    
    parser.add_argument('-m','--models',
                        type=int,
                        default= [
                        ("aarima",AutoARIMA()),
                        ("exp",ExponentialSmoothing()),
                        ("fft",FFT(required_matches=set(), nr_freqs_to_keep=None)),
                        ("nbeats", NBEATSModel(
                            layer_widths=300,
                            input_chunk_length=8,
                            output_chunk_length=1,
                            num_blocks=3,
                            generic_architecture=False,
                            force_reset=True,
                            nr_epochs_val_period=5,
                            n_epochs=100)),
                        # ("rndforest",RandomForest(
                        #     lags=[-1,-2,-3], 
                        #     lags_past_covariates=[-1,-2,-3])),
                        # ("regression",RegressionModel(
                        #     lags=[-1,-2,-3,-4], 
                        #     lags_past_covariates=[-1,-2,-3,-4])),
                        # ("brnn",BlockRNNModel(
                        #     model='LSTM',
                        #     input_chunk_length=10,
                        #     output_chunk_length=1,
                        #     n_epochs=200,
                        #     model_name='Brnn',
                        #     force_reset=True)),
                        ("rnn",RNNModel(
                            model='LSTM',
                            hidden_dim=30,
                            dropout=0.2,
                            batch_size=32,
                            n_epochs=100,
                            optimizer_kwargs={'lr': 1e-3},
                            model_name='RNN_model',
                            log_tensorboard=True,
                            training_length=40,
                            input_chunk_length=8,
                            force_reset=True,
                            likelihood=GaussianLikelihood()
                        )),
                        # ("tcn",TCNModel(
                        #     dropout=0.2,
                        #     batch_size=32,
                        #     n_epochs=200,
                        #     optimizer_kwargs={'lr': 1e-3},
                        #     dilation_base=2,
                        #     num_layers=2,
                        #     random_state=0,
                        #     input_chunk_length=6,
                        #     output_chunk_length=1,
                        #     kernel_size=4,
                        #     model_name="tcn_model",
                        #     num_filters=6,
                        #     likelihood=GaussianLikelihood(),
                        #     force_reset=True
                        # )),
                        # ("gennbeats", NBEATSModel(
                        #     layer_widths=300,
                        #     input_chunk_length=8,
                        #     output_chunk_length=1,
                        #     generic_architecture=True,
                        #     num_blocks=3,
                        #     num_layers=4,
                        #     num_stacks=5,
                        #     force_reset=True,
                        #     n_epochs=100)),
                        # ("transformer",TransformerModel(
                        #     input_chunk_length=8,
                        #     output_chunk_length=1,
                        #     batch_size=32,
                        #     n_epochs=100,
                        #     model_name='transformer',
                        #     d_model=16,
                        #     nhead=4,
                        #     likelihood=GaussianLikelihood(),
                        #     num_encoder_layers=3,
                        #     num_decoder_layers=3,
                        #     dim_feedforward=32,
                        #     dropout=0.2,
                        #     activation="relu",
                        #     force_reset=True,
                        #     log_tensorboard=True
                        #     )),
                        ],
                        required=False,
                        help="Forecast horizon.")
    
    
    args = parser.parse_args("")

    # teams
    teams = ['crd', 'atl', 'rav', 'buf', 'car', 'chi', 'cin', 'cle', 'dal',
        'den', 'det', 'gnb', 'htx', 'clt', 'jax', 'kan', 'rai', 'sdg',
        'ram', 'mia', 'min', 'nwe', 'nor', 'nyj', 'phi', 'pit', 'sfo',
        'sea', 'tam', 'oti', 'was','nyg']

    start = args.start
    end = args.end
    fh = args.horizon 
      
    forecasters = args.models
    f = forecasters[0]
     
    teamdata = read_data(teams, [start,end])
    d = teamdata[0]
    

# %%
    aa = AutoARIMA()
    
    X = d[['Off_Totyd','Off_TO','Def_TO','PassY/A', 'PassCmp', 'RushAtt',
                'RushYds', 'RushY/A', 'FGM', 'FGA', 'Pnt', 'PntYds']]
    y = d['TmScore']
    X_mu = X.mean(axis=0)
    X = X.append(X_mu, ignore_index=True)
    
    y_ts=TimeSeries.from_series(y)
    X_ts=TimeSeries.from_dataframe(X)
    
    all_stats = None
    for col in X_ts.columns:
        if all_stats is None:
            all_stats = X_ts[col]
        else:
            all_stats = X_ts.stack(X_ts[col])
    
    aa.fit(y_ts,all_stats)
    aa.predict(1,all_stats)
    
    
# %%
    forecasts, models = get_forecasts(teamdata, forecasters, fh)
    forecasts


# %%
    #ensembles = get_ensemble_forecasts(teamdata, forecasters, fh)
    #ensembles


