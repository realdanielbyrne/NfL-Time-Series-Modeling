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
from patsy import dmatrices
from darts import TimeSeries
from darts.metrics import mape, smape, mase
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
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

    stats['OT'] = np.where((stats['OT']=='OT'), 1, 0)
    stats['@'] = np.where((stats['@']=='@'), 1, 0)
    stats['W/L'] = np.where((stats['W/L']=='W'), 1, 0)

    stats['Def_TO'] = stats['Def_TO'].fillna(0)
    stats['Off_TO'] = stats['Off_TO'].fillna(0)
    

    stats.index = pd.RangeIndex(len(stats.index))
    stats.index = range(len(stats.index))  
    stats['Team'] = team


    return stats

def get_ts_data(teamdata, add_mean = True, target_col = 'TeamScore', covs = None):

    # Split target and covariates     
    target = teamdata[target_col]
    # covs = teamdata[[
    #     'Off_1stDn', 'Off_Totyd', 'Off_PassYd', 'Off_RushYd','TeamScore','Off_TO',
    #     'Def_1stDn', 'Def_Totyd', 'Def_PassYd', 'Def_RushYd','@']]

    covs = teamdata[[
        'Off_1stDn', 'Off_Totyd', 'Off_PassYd', 'Off_RushYd','TeamScore','OT','@',
        'Def_1stDn', 'Def_Totyd', 'Def_PassYd', 'Def_RushYd','Def_TO']]
    covs = covs.astype(np.float32)
    covs = covs.drop(target_col, axis = 1)

    # Adds a simulated sample to build look ahead models
    if add_mean:
        mu = covs.mean(axis=0)
        covs = covs.append(mu,ignore_index=True)
        target.loc[len(target.index)] = target.mean(axis=0)
        
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
    
    ret = { "target":target_ts, "cov":all_stats, "team":teamdata['Team'] }

    return ret

def get_parser():
    parser = argparse.ArgumentParser(description='NFL spread Predictor.')
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

    parser.add_argument('--target',
                        default='TeamScore',                        
                        required=False,
                        help="The target stat to predict.")                        


    return parser

def get_prediction(p1, p2, team1, team2, modelname):
    # Calculate predictions
    spread = -round(p1 - p2, 0)
    ou   = round(p1 + p2, 0)
        
    prediction = {
        team1: round(p1,0), 
        team2: round(p2,0), 
        "spread": spread, 
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

def brnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):
      
    model_cov = BlockRNNModel(
        model='LSTM', 
        input_chunk_length=10, 
        output_chunk_length=1, 
        n_epochs=300,
        model_name='Brnn',
        force_reset=True)

    model_cov.fit([train1,train2],
             past_covariates=[cov1,cov2],
             verbose=True)
    
    p = model_cov.predict(1, series=[train1,train2], past_covariates = [cov1, cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2, team1, team2, "BlockRNNModel")

    return pred

def rnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):
    
    rnn = RNNModel(
        model='LSTM',
        hidden_dim=30,
        dropout=0.2,
        batch_size=32,
        n_epochs=200,
        optimizer_kwargs={'lr': 1e-3}, 
        model_name='RNN_model',
        log_tensorboard=True,
        training_length=40,
        input_chunk_length=8,
        force_reset=True,
        likelihood=GaussianLikelihood()
    )

    rnn.fit([train1,train2],
             future_covariates=[cov1,cov2],
             verbose=True)
    
    p = rnn.predict(1, series=[train1,train2], future_covariates = [cov1, cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2, team1, team2, "RNNModel")

    return pred

def fft_model(train1,train2,scale1,scale2,team1,team2):

    fft_model1 = FFT(trend='poly')
    fft_model1.fit(train1)    
    p1 = fft_model1.predict(1)

    fft_model2 = FFT(trend='poly')
    fft_model2.fit(train2)    
    p2 = fft_model2.predict(1)

    p1 = scale1.inverse_transform(p1).first_value()
    p2 = scale2.inverse_transform(p2).first_value()

    pred = get_prediction(p1, p2, team1, team2, "FFT")

    return pred

def tcn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):
    deeptcn = TCNModel(
        dropout=0.2,
        batch_size=32,
        n_epochs=400,
        optimizer_kwargs={'lr': 1e-3}, 
        dilation_base=2,
        num_layers = 2,
        random_state=0,
        input_chunk_length=6,
        output_chunk_length=1,
        kernel_size=4,
        model_name= "tcn_model",
        num_filters=6,
        likelihood=GaussianLikelihood(),
        force_reset=True
        )

    deeptcn.fit([train1,train2],
             past_covariates=[cov1,cov2],
             verbose=True)
    
    p = deeptcn.predict(1, series=[train1,train2], past_covariates = [cov1, cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()

    pred = get_prediction(p1, p2, team1, team2, "TCNModel") 

    return pred

def beats_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2, scale = True):

    beatsmodel = NBEATSModel(
        layer_widths = 300,
        input_chunk_length = 8,
        output_chunk_length = 1, 
        num_blocks = 3,
        generic_architecture = False,
        force_reset = True,
        nr_epochs_val_period = 5,
        n_epochs = 200)
    
    beatsmodel.fit( series = [train1, train2], 
                    past_covariates = [cov1, cov2], 
                    verbose = True)

    p = beatsmodel.predict(1, series = [train1, train2], past_covariates = [cov1,cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2,team1, team2, "NBEATSModel")

    return pred
 
def beats_model_generic(train1,train2,cov1,cov2,scale1,scale2,team1,team2, scale = True):
    beatsmodel = NBEATSModel(
        layer_widths=300,
        input_chunk_length=8,
        output_chunk_length=1, 
        generic_architecture=True,
        num_blocks = 3,
        num_layers = 4,
        num_stacks = 5,        
        force_reset = True,
        n_epochs=200)
    
    beatsmodel.fit([train1,train2],past_covariates = [cov1,cov2],verbose = True)
    p = beatsmodel.predict(1, series = [train1,train2], past_covariates = [cov1,cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2,team1, team2, "NBEATS Generic")

    return pred   

def transformer_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):    
    trans_model = TransformerModel(
        input_chunk_length = 8,
        output_chunk_length = 1,
        batch_size = 32,
        n_epochs = 200,
        model_name = 'transformer',    
        d_model = 16,
        nhead = 4,
        likelihood=GaussianLikelihood(),
        num_encoder_layers = 3,
        num_decoder_layers = 3,
        dim_feedforward = 32,
        dropout = 0.2,
        activation = "relu",
        force_reset=True,
        log_tensorboard = True
    )

    trans_model.fit(series=[train1, train2], past_covariates = [cov1, cov2], verbose=True)
    p = trans_model.predict(1, series=[train1, train2], past_covariates = [cov1, cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2, team1, team2, "TransformerModel")     

    return pred

def randomforest_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):
    
    randf_model = RandomForest(lags=[-1,-2,-3],lags_past_covariates=[-1,-2,-3])

    randf_model.fit(series=[train1,train2], past_covariates = [cov1,cov2])
    p = randf_model.predict(1, series=[train1,train2], past_covariates = [cov1,cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2, team1, team2, "RandomForest")     

    return pred

def regression_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2):
    
    regression_model = RegressionModel(lags=[-1,-2,-3,-4],lags_past_covariates=[-1,-2,-3,-4])

    regression_model.fit(series=[train1,train2], past_covariates = [cov1,cov2])
    p = regression_model.predict(1, series=[train1,train2], past_covariates = [cov1,cov2])
    p1 = scale1.inverse_transform(p[0]).first_value()
    p2 = scale2.inverse_transform(p[1]).first_value()
    pred = get_prediction(p1, p2, team1, team2, "Regression") 

    return pred

def ensemble_model(preds):
    preds_df = pd.DataFrame(preds)
    mu = round(preds_df.mean(axis=0),0)
    mu['model'] = 'mu'
    preds_df = preds_df.append(mu,ignore_index=True)

    return preds_df

def split_and_scale_data(t1data,t2data):
    train1 = t1data['target'][:-1]
    cov1 = t1data['cov']

    train2 = t2data['target'][:-1]
    cov2 = t2data['cov']

    # scale training data
    scale1 = Scaler()
    train1 = scale1.fit_transform(train1)    
    scale2 = Scaler()
    train2 = scale2.fit_transform(train2)    

    # scale covariates    
    cov1 = Scaler().fit_transform(cov1)        
    cov2 = Scaler().fit_transform(cov2)

    return train1, train2, cov1, cov2, scale1, scale2

if __name__ == "__main__":

    # Parameters
    parser = get_parser()
    args = parser.parse_args("")
    teams = []
    
    #teams.append(['atl','nor'])
    #teams.append(['den','dal'])
    #teams.append(['htx','mia'])
    #teams.append(['min','rav'])
    #teams.append(['cle','cin'])
    teams.append(['kan','gnb'])

    target_col = 'TeamScore'
    years=['2018','2019','2020','2021'] 
    
    matchups = {}
    # Get Data
    for t in teams:
        team1 = t[0]
        team2 = t[1]
        add_mean = True
        t1data, t2data = getMatchupData(team1, team2, years, add_mean, target_col)
        train1, train2, cov1, cov2, scale1, scale2 = split_and_scale_data(t1data, t2data)

        # Plot
        # t1data['target'].plot(label=team1)
        # t2data['target'].plot(label=team2)
        # plt.ylabel(target_col)
        # plt.show()

        # t1data['target'][-5:].plot(label=team1)
        # t2data['target'][-5:].plot(label=team2)
        # plt.ylabel(target_col)
        # plt.show()

        # Predict
        preds = []
        preds.append (arima_model(t1data, t2data, team1, team2))
        preds.append (fft_model(train1,train2,scale1,scale2,team1,team2))
        preds.append (randomforest_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
        preds.append (exp_model(t1data, t2data, team1, team2))
        preds.append (brnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
        preds.append (regression_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
        preds.append (rnn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
        preds.append (tcn_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
        preds.append (beats_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))
        preds.append (beats_model_generic(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
        preds.append (transformer_model(train1,train2,cov1,cov2,scale1,scale2,team1,team2))        
        preds_df = ensemble_model(preds)
        
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
        
    