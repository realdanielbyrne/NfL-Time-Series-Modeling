
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
import datetime
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

