# NFL Time Series Modeling to Predict Betting Lines

If only we could predict the future of any future sporting event... We could be rich!

![Greys Sports Almanac](./almanac.jpg)

## Quick Start

1. Register for a 7 day free trial to (Pro Football API)[https://profootballapi.com/signup] to receive an API key.
2. Source this file and execute getPredictions('DAL','CHI') on the r command line

    getPredictions('DAL','CHI', 2019)


## Motivation

- To Win!
- Gamblers on average win between 53% and 55% of the time.
- Bettors make ~$93,000 a year

Even a +1%  can mean yield $$$ depending on bets made and amount wagered.  


## Traditional Sports Modeling

Traditional methods rely on moving averages

- Dak Prescott’s season completion% in 2016, 2018, and 2019 are within +/-1% of his career average.
- Players and Teams tend to revert to their mean

Vegas makes modifications to predictions for Home/Away, Weather, and Travel, etc to help improve these mean based predictions.  However, they are wrong 45 - 47% of the time.

![Dak's Stats](./dakstats.png)

## Cyclical patterns in Human Performance

![Dak's Completion % 2018-20180](./dakcmp.png)

- The wandering and cyclic behaviors evident in stats are a mirror into real life
- Good players have a higher mean than average players
- When all players on a team have a good game, that can raise the expectation of the team winning

## Stationary Time Series

Stats resemble stationary time series.

- Mean and variance do not dependent on time
- Correlation of two points `t1 and t2` depends only on how far apart and not where they are in the time series.

Thus the conditions of stationarity are generally met.

![Dallas Cowboys Score 2018-2018](./dalgamscores.png)


