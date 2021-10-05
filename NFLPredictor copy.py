from darts.utils.utils import T
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import argparse
warnings.simplefilter(action='ignore', category=FutureWarning)

from darts import TimeSeries
from darts.utils.timeseries_generation import gaussian_timeseries, linear_timeseries, sine_timeseries
from darts.metrics import mape, smape, mase
from darts.dataprocessing.transformers import Scaler
from darts.utils.timeseries_generation import datetime_attribute_timeseries
from darts.models import (
    Prophet,
    ExponentialSmoothing,
    ARIMA,
    AutoARIMA,
    RegressionEnsembleModel,
    RegressionModel,
    Theta,
    FFT,
    NBEATSModel,
    RNNModel,
    TCNModel,
    TransformerModel,
    BlockRNNModel
)

baseurl = 'https://www.pro-football-reference.com/teams/{team}/{year}.htm'
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
        stats = stats.append(df)

    stats.fillna(0)
    stats.index = pd.RangeIndex(len(stats.index))
    stats.index = range(len(stats.index))  
    stats['Team'] = team

    return stats

def getMatchupData (team1 = 'dal', team2 = 'phi', years=['2018','2019','2020','2021'], add_mean = True):
    
    t1stats = getTeamGameStats(team1, years)
    t2stats = getTeamGameStats(team2, years)

    t1_ts = get_ts_data(t1stats, add_mean)
    t2_ts = get_ts_data(t2stats, add_mean)

    teams = t1_ts, t2_ts
    return teams

def get_ts_data(teamdata, add_mean = True):

    # Split target and covariates     
    target = teamdata['TeamScore']
    covs = teamdata[[
        'Off_1stDn', 'Off_Totyd', 'Off_PassYd', 'Off_RushYd',
        'Def_1stDn', 'Def_Totyd', 'Def_PassYd', 'Def_RushYd']]

    # Adds a simulated sample to build look ahead models
    if add_mean:
        mu = covs.mean(axis=0)
        covs = covs.append(mu,ignore_index=True)
        target.loc[len(target.index)] = target.mean(axis=0)
        
    # Get TimeSeries
    target_ts = TimeSeries.from_series(target)
    covs_ts = TimeSeries.from_dataframe(covs)

    # Normalize data    
    off_stats = covs_ts['Off_1stDn'].stack(covs_ts['Off_Totyd']).stack(covs_ts['Off_PassYd'].stack(covs_ts['Off_RushYd']))
    def_stats = covs_ts['Def_1stDn'].stack(covs_ts['Def_Totyd']).stack(covs_ts['Def_PassYd'].stack(covs_ts['Def_RushYd']))
    all_stats = off_stats.stack(def_stats)

    ret = { "target":target_ts, "cov":off_stats, "team":teamdata['Team'] }
    return ret

def get_parser():
    parser = argparse.ArgumentParser(description='NFL Line Predictor.')
    parser.add_argument('--team1',
                        action='store',
                        type=str,
                        default="dal",
                        required=False,
                        help="The first team in a matchup.")

    parser.add_argument('--team2',
                        type=str,
                        default='phi',                        
                        required=False,
                        help="The second team in a matchup.")

    parser.add_argument('--years',
                        default=['2018','2019','2020','2021'],
                        nargs='?',                        
                        required=False,
                        help="The second team in a matchup.")                        

    return parser

def get_prediction(p1, p2, team1, team2, modelname):
    # Calculate predictions
    line = -round(p1 - p2, 0)
    ou   = round(p1 + p2, 0)
        
    prediction = {
        team1: p1, 
        team2: p2, 
        "line": line, 
        "ou": ou,
        "model":modelname 
        }
    return prediction

def arima_model(t1data, t2data, team1, team2):

    model1 = AutoARIMA()
    train1 = t1data['target'][:-1] 
    cov1, cval1 = t1data['cov'][:-1], t1data['cov'][-1:]

    scale1 = Scaler()
    train1 = scale1.fit_transform(train1)
    
    covscaler1 = Scaler()
    cov1 = covscaler1.fit_transform(cov1)
    cval1 = covscaler1.transform(cval1)

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

def exp_model(t1data, t2data, team1, team2):
    model1 = ExponentialSmoothing()
    model1.fit(t1data['target'][:-1])
    p1 = model1.predict(1).first_value()

    model2 = ExponentialSmoothing()
    model2.fit(t2data['target'][:-1])
    p2 = model2.predict(1).first_value()

    pred = get_prediction(p1, p2, team1, team2, "ExponentialSmoothing")
    return pred

def beats_model(t1data, t2data, team1, team2):
    beatsmodel = NBEATSModel(
        input_chunk_length=5,
        output_chunk_length=1, 
        generic_architecture=True,
        n_epochs=20)
    
    train1 = t1data['target'][:-1]     
    cov1 = t1data['cov']
    train2 = t2data['target'][:-1]
    cov2 = t2data['cov']    

    beatsmodel.fit([train1,train2],past_covariates = [cov1,cov2],verbose = True)
    p = beatsmodel.predict(1,series=[train1,train2], past_covariates=[cov1,cov2])
    pred = get_prediction(p[0].first_value(),p[1].first_value(),team1,team2,"NBEATSModel")
    print(pred)

    return pred


if __name__ == "__main__":

    # Parameters
    parser = get_parser()
    args = parser.parse_args("")
#    team1 = args.team1
#    team2 = args.team2

    team1 = 'nwe'
    team2 = 'tam'


    years=['2018','2019','2020','2021']
    
    # Get Data
    add_mean = False
    t1data, t2data = getMatchupData(team1, team2, years, add_mean)

    # Visualize
    t1data['target'].plot(label=team1)
    t2data['target'].plot(label=team2)
    plt.show()

    preds = []
    preds.append (arima_model(t1data, t2data, team1, team2))
    preds.append (exp_model(t1data, t2data, team1, team2))
    #preds.append (beats_model(t1data, t2data, team1, team2))

    for p in preds:
        print('{team1}: {forecast1:.0f} \t{team2}: {forecast2:.0f} \tLine: {line:+}\tOU: {ou:.0f} \t{name:>}'
        .format(                
            team1=team1.upper(), forecast1=p[team1],
            team2=team2.upper(), forecast2=p[team2],
            line=p['line'], ou=p['ou'],
            name=p['model']))


