import requests
import pandas as pd
import darts
import numpy as np
import torch
import matplotlib.pyplot as plt


from darts import TimeSeries
from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries
from darts.metrics import mape, smape
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import RNNModel, TCNModel, TransformerModel, NBEATSModel, BlockRNNModel
from darts.models import (
    NaiveSeasonal,
    NaiveDrift,
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT
)


baseurl = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
def getTeamGameStats  (team='dal',years=['2019','2020','2021']):
 
    stats = pd.DataFrame()
    for year in years:
        url = baseurl.format(team=team,year=year)
        html = requests.get(url).content
        df_list = pd.read_html(html)
        df = df_list[1]
        df.columns = ["_".join(a) for a in df.columns.to_flat_index()]

        df.columns = ['Week','Day','Date','Time','BoxScore',
            'W/L','OT','Record','@','Opponent','TeamScore',
            'OpponentScore','Off_1stDn','Off_Totyd','Off_PassYd',
            'Off_RushYd','Off_TO','Def_1stDn','Def_Totyd',
            'Def_PassYd','Def_RushYd','Def_TO','ExpOff',
            'ExpDef','ExpSpTeams']
        df = df.dropna(subset=['W/L'])
        df = df[['TeamScore',
            'OpponentScore','Off_1stDn','Off_Totyd','Off_PassYd',
            'Off_RushYd','Off_TO','Def_1stDn','Def_Totyd',
            'Def_PassYd','Def_RushYd','Def_TO']]
        
        
        
        stats = stats.append(df)

    stats.fillna(0)
    stats.index = pd.RangeIndex(len(stats.index))
    stats.index = range(len(stats.index))  
    return stats

def getMatchupData (team1 = 'dal', team2 = 'chi', years=['2019','2020','2021'], target_col = 'TeamScore'):
    t1data = getTeamGameStats(team1, years)
    t2data = getTeamGameStats(team2, years)

    t1data = get_ts_data(t1data, target_col)
    t2data = get_ts_data(t2data, target_col)

    teams = [t1data, t2data]
    return teams

def get_ts_data(teamdata, target_col='TeamScore'):
    
    # Split target and covariates     
    target = teamdata[target_col]
    covs = teamdata[[
        'Off_1stDn', 'Off_Totyd', 'Off_PassYd', 'Off_RushYd',
        'Def_1stDn', 'Def_Totyd', 'Def_PassYd', 'Def_RushYd']]

    # Get TimeSeries
    target_ts = TimeSeries.from_series(target)
    covs_ts = TimeSeries.from_dataframe(covs)

    # Normalize data    
    offensive_stats = covs_ts['Off_1stDn'].stack(covs_ts['Off_Totyd']).stack(covs_ts['Off_PassYd'].stack(covs_ts['Off_RushYd']))
    defensive_stats = covs_ts['Def_1stDn'].stack(covs_ts['Def_Totyd']).stack(covs_ts['Def_PassYd'].stack(covs_ts['Def_RushYd']))

    ret = { "target":target_ts, "off":offensive_stats, "def":defensive_stats }
    return ret

def arima_model(t1data, t2data, include_covariates = True, scale = False):
    model = AutoARIMA()
    t1forecast = eval_model(model, t1data, include_covariates, scale)
    t2forecast = eval_model(model, t2data, include_covariates, scale)
    

def eval_model(model, teamdata, covs, include_covariates = True, scale = True):
    
    train, val = teamdata['target'][:-1], teamdata['target'][-1:]
    covs = teamdata['off']

    if scale:
        scaler= Scaler()
        train = scaler.fit_transform(train)
        val = scaler.fit_transform(val)
        cov_scaler = Scaler()
        covs = cov_scaler.fit_transform(covs)
    
    if include_covariates:
        model.fit(train, future_covariates=covs[:-1])
        forecast = model.predict(len(val),covs[-1:])
    else:
        model.fit(train)
        forecast = model.predict(len(val))

    if scale:
        forecast = scaler.inverse_transform(forecast)
        val = scaler.inverse_transform(val)

    print('model {} obtains MAPE: {:.2f}%'.format(model, mape(val, forecast)))
    print('forecast :',forecast.first_value())
    print('val :',val.first_value())

team1='dal'
team2='chi'

t1data,t2data = getMatchupData(team1,team2)
arima =  arima_model()

model = AutoARIMA()
model.fit(train,future_covariates=cov_train)
forecast = model.predict(len(val),all_stats[-1:],num_samples=20)
forecast = transformer.inverse_transform(forecast)

val = transformer.inverse_transform(val)
print('model {} obtains MAPE: {:.2f}%'.format(model, mape(val, forecast)))
print('forecast :',forecast.first_value())
print('val :',val.first_value())

