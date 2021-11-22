from sktime.datasets import load_airline
from sktime.forecasting.naive import NaiveForecaster
from sktime.utils.plotting import plot_series
import pandas as pd

team = 'dal'
start = '2016'
stop = '2021'

# step 1: data specification
filename = './stats/combinedstats/{team}_{start}_{stop}.csv'.format(team=team,start=start,stop=stop)
y = pd.read_csv(filename)

# step 2: specifying forecasting horizon
fh = 1

# step 3: specifying the forecasting algorithm
forecaster = NaiveForecaster(strategy="last")

# step 4: fitting the forecaster
forecaster.fit(y['TmScore'])

# step 5: querying predictions
y_pred = forecaster.predict(fh)

# optional: plotting predictions and past data
plot_series(y['TmScore'], y_pred, labels=["y", "y_pred"])