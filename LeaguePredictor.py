from darts.models.forecasting.regression_model import RegressionModel
from darts.utils.utils import T
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import statsmodels.api as sm
import datetime
from darts import TimeSeries
from darts.metrics import mape, smape, mase
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from models import *
from darts.models import (
    ExponentialSmoothing,
    AutoARIMA,
    RegressionModel,    
    RandomForest,
    FFT,
    NBEATSModel,
    RNNModel,
    TCNModel,
    TransformerModel,
    BlockRNNModel
)
import warnings
warnings.filterwarnings("ignore")
import logging
logging.disable(logging.CRITICAL)


baseurl = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
teams = ['crd','atl','rav','buf','car','chi','cin','cle','dal',
         'den','det','gnb','htx','clt','jax','kan','rai','sdg',
         'ram','mia','min','nwe','nor','nyj','phi','pit','sfo',
         'sea','tam','oti','was']

def getMatchupData (team1 = 'dal', 
                    team2 = 'phi', 
                    years = ['2018','2019','2020','2021'], 
                    add_mean = True, 
                    target_col='TeamScore',
                    covs = None):
    
    t1stats = getTeamGameStats(team1, years)
    t2stats = getTeamGameStats(team2, years)

    t1_ts = get_ts_data(t1stats, add_mean, target_col)
    t2_ts = get_ts_data(t2stats, add_mean, target_col)

    teams = t1_ts, t2_ts
    return teams


def arima_model(train,  team1, team2):
    
    model = AutoARIMA()

    model1.fit(train1, future_covariates=cov1)
    p1 = model1.predict(1, future_covariates=cval1)
    p1 = scale1.inverse_transform(p1).first_value()
 
    model2 = AutoARIMA()
    train2 = t2data['target'][:-1]
    cov2, cval2 = t2data['cov'][:-1], t2data['cov'][-1:]

    scale2 = Scaler()
    train2 = scale2.fit_transform(train2)
    
    covscaler2 = Scaler()
    cov2 = covscaler2.fit_transform(cov2)
    cval2 = covscaler2.transform(cval2)

    model2.fit(train2, future_covariates=cov2)
    p2 = model2.predict(1, future_covariates=cval2)
    p2 = scale2.inverse_transform(p2).first_value()

    pred = get_prediction(p1, p2, team1, team2, "AutoARIMA")
    return pred


def getTeamsStats(years):
    team_stats = {}
    for team in teams:
        stats = getTeamGameStats(team, years=['2018','2019','2020','2021'])
        team_stats[team] = stats

    return team_stats

def getTeamGameStats  (team='dal',years=['2018','2019','2020','2021']):
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
        df['year'] = year
        stats = stats.append(df)

    stats['OT'] = np.where((stats['OT']=='OT'), 1, 0)
    stats['@'] = np.where((stats['@']=='@'), 1, 0)
    stats['W/L'] = np.where((stats['W/L']=='W'), 1, 0)

    stats['Def_TO'] = stats['Def_TO'].fillna(0)
    stats['Off_TO'] = stats['Off_TO'].fillna(0)
    stats['Team'] = team

    stats[['month','day']] = stats.Date.str.split(expand=True)
    stats['Timestamp_str'] = stats['year'].str.cat(stats['month'], sep ="/").str.cat(stats['day'], sep ="/")
    stats['Timestamp'] = pd.to_datetime(stats['Timestamp_str'], format="%Y/%B/%d")

    #stats.index = stats['Timestamp']
    stats.index = pd.RangeIndex(len(stats.index))
    stats.index = range(len(stats.index))  

    return stats

def getTeamsTimeSeries(team_stats, add_mean = True, target_col = 'TeamScore'):
    team_ts_stats = {}
    for team in team_stats:
        teamdata = team_stats[team]
        ts = get_ts_data(teamdata, add_mean, target_col)
        team_ts_stats[team] = ts
    return team_ts_stats

def get_ts_data(teamdata, add_mean = True, target_col = 'TeamScore', scale = True):
    target = teamdata[target_col]
    covs = teamdata[[
        'Off_1stDn', 'Off_Totyd', 'Off_PassYd', 'Off_RushYd','TeamScore','OT','@',
        'Def_1stDn', 'Def_Totyd', 'Def_PassYd', 'Def_RushYd','Def_TO']]
    covs = covs.astype(np.float32)
    covs = covs.drop(target_col, axis = 1)

    # Adds a simulated sample to build look ahead models
    if add_mean:
        mu = covs.mean(axis=0)
        covs = covs.append(mu,ignore_index=True) 
        target.loc[len(target)] = target.mean(axis=0)
        
    # Get TimeSeries
    target_ts = TimeSeries.from_series(target)
    covs_ts = TimeSeries.from_dataframe(covs)

    # stack covariates    
    all_stats = None
    for col in covs.columns:
        if all_stats is None:
            all_stats = covs_ts[col]
        else:
            all_stats = covs_ts.stack(covs_ts[col])            
    
    train_ts = target_ts[:-1]

    if scale:
        # scale training data
        scaler = Scaler()
        train_ts = scaler.fit_transform(train_ts) 
    else:
        scaler: None
    
    return { "train":train_ts, "covs":covs_ts, "scaler":scaler }

def get_parser():
    parser = argparse.ArgumentParser(description='NFL spread Predictor.')
    parser.add_argument('--years',
                        default=['2018','2019','2020','2021'],
                        nargs='?',                        
                        required=False,
                        help="The second team in a matchup.")                        

    parser.add_argument('--target',
                        default='TeamScore',                        
                        required=False,
                        help="The target stat to predict.")     

    parser.add_argument('--add_mean',
                    default=True,                        
                    required=False,
                    help="Add a fake entry at the end of the time series equivalent to the mean on axis=0.")                      

    parser.add_argument('-s,--scale',
                    default='scale',                        
                    required=False,
                    help="Scale the training data.") 

    args = parser.parse_args("")
    return args

if __name__ == "__main__":

    # Parameters
    args = get_parser()

    years = args.years
    target_col = args.target
    add_mean = args.add_mean
    scale = True

    # Get Data
    team_stats = getTeamsStats(years)
    team_ts_stats = getTeamsTimeSeries(team_stats, add_mean, target_col )
    
    arimamodel = arima_model(team_ts_stats)
    
    # Predict
    # preds = []
    # preds.append (arima_model(t1data, t2data, team1, team2))
    # preds.append (fft_model(train1,train2,scale1,scale2,team1,team2))
    # preds.append (randomforest_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
    # preds.append (exp_model(t1data, t2data, team1, team2))
    # preds.append (brnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
    # preds.append (regression_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
    # preds.append (rnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
    # preds.append (tcn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
    # preds.append (beats_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
    # preds.append (beats_model_generic(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
    # preds.append (transformer_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
    # preds_df = ensemble_model(preds)
    
    # plot regression
    matchup = "{} ~ {}".format(team1,team2)
    # y, X = dmatrices(matchup, data=preds_df, return_type='dataframe')
    # mod = sm.OLS(y, X)    # Describe model
    # res = mod.fit()       # Fit model        
    
    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.plot(X[team2], y, "o", label="data")
    # ax.plot(X[team2], res.fittedvalues, "r--.", label="OLS")
    # ax.legend(loc="best")
    # ax.set_xlabel(team1)
    # ax.set_ylabel(team2)
    # plt.show()        
    matchups[matchup]=preds
    print(preds_df)
    
    