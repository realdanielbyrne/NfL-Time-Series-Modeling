# Title : NFLLinePredictor.r
#
# Descrioption: Scrapes data from profootballapi.com
# Get an API keyfrom https://profootballapi.com/

#Require the package so you can use it
require("httr")
require("jsonlite")
require("rvest")
require('mosaic')
require('tswge')
require('vars')
require('nnfor')
require("plyr")
require("vars")
require("GGally")

# Temporary Key (get your free trial key at)
getApiKey <- function() {
  api_key = "D15KXhxEp7uqUJesQL29ZioPA6aYjgb0"  # replace with your API key
  return (api_key)
}

# Returns base API url
getUri <- function() {
  uri = 'https://profootballapi.com/'
  return (uri)
}

# https://profootballapi.com/docs/teams
getTeamStats <- function(team='dal', year) 
{
  if (missing(year))
  {
    query = list(api_key = getApiKey(), team = team, season_type = 'REG' )
  }
  else
  {
    query = list(api_key = getApiKey(), team = team, year = year, season_type = 'REG' )
  }

  url = paste0(getUri(),'teams')
  return (request(url,query))
}

# https://profootballapi.com/docs/schedule
getScheduleStats <- function(year)
{
  if(missing(year))
  {
    query = list(api_key = getApiKey(), season_type = 'REG')
  }
  else
  {
    query = list(api_key = getApiKey(), year = year,season_type = 'REG')
  }
  
  url = paste0(getUri(),'schedule')
  return (request(url,query))
}

# https://profootballapi.com/docs/players
getPlayerStats <- function(player_name='', stats_type = 'passing', year = '') 
{
  query = list(api_key = getApiKey(), player_name = player_name, stats_type = stats_type, year = year, season_type = 'REG')

  url = paste0(getUri(),'players')
  return (request(url,query))
}

# Base api request
request <- function(url,query)
{
  resp = POST(url, query = query)
  txt = content(resp, as="text")
  results = as.data.frame(fromJSON(txt),simplifyDataFrame = TRUE)
  return (results)
}

# Combines Schedule statistics with Team Statistics 
# Removes columns with little or no value
combineScheduleWithStats <- function(sched, teamStats, team) 
{
  teamStats$id = teamStats$nfl_game_id
  all = inner_join(teamStats,sched,by ='id')
  all$gamedate = substring(all$nfl_game_id,1,8)
  all$gamedate = as.Date(all$gamedate, "%Y%m%d")

  all$isHome = as.integer(as.logical(all$home == team))
  all$won = as.integer(ifelse(all$isHome == 1,all$home_score > all$away_score,all$away_score > all$home_score))
  all = all %>% dplyr::select(-season_type,-home,-away, -nfl_game_id, -final,-team, -year,-day,-month,-time)
  all = all %>% dplyr::select(id,everything())
  all$score = ifelse(all$isHome,all$home_score,all$away_score)
  return (all)
}

# Get Teams Stats 
getTeamData <- function(team = 'DAL', year = 2019)
{
  lastYear = year-1
  schedyear = getScheduleStats(year)
  schedprioryear = getScheduleStats(lastYear)
  sched = rbind(schedyear,schedprioryear)
  
  teamStats = getTeamStats(team, year)
  teamStatsPrior = getTeamStats(team, lastYear)
  teamStats = rbind (teamStats,teamStatsPrior)
  
  teamStats$trnovr1 = dplyr::lag(teamStats$trnovr,1)
  teamStats$penyds1 = dplyr::lag(teamStats$penyds,1)
  teamStats$totyds1 = dplyr::lag(teamStats$totyds,1)
  teamStats$ptyds1  = dplyr::lag(teamStats$ptyds,1)

  all = combineScheduleWithStats(sched, teamStats, team)
  return (all)
}

# Gets complete stats for both teams
getMatchupData <- function(team1 = 'DAL', team2 = 'CHI', year = 2019)
{
  t1data = getTeamData(team1, year)
  t2data = getTeamData(team2, year)
  teams = list(t1data,t2data) 
  return(teams)
}

# Builds MLP model for both teams.
# Models score as a predictor for team performance
# Returns the model predictions for each team, the difference,the sum and the ASE
mlp_model <- function(teams, team1,team2,autolag = FALSE)
{

  # Team 1 model
  t1 = teams[[1]]
  score1 = t1$score
  l = length(t1$score)-1 
  score_ts1 = ts(score1[1:l])
  t1_2 = t1[1:l,]
  
  t1xregs = data.frame(totyds = ts(t1_2$totyds), 
                       trnovr = ts(t1_2$trnovr), 
                       penyds = ts(t1_2$penyds),
                       ptyds  = ts(t1_2$ptyds))
  if (autolag) {
    team1_fit = mlp(score_ts1, xreg = t1xregs, hd=c(8,4,2), sel.lag=TRUE, hd.auto.type = "elm", reps = 100, xreg.lags = list(1,1,1,1))
  }
  else {
    team1_fit = mlp(score_ts1, xreg = t1xregs, hd=c(8,4,2), sel.lag=TRUE, hd.auto.type = "elm", reps = 100, xreg.lags = list(1,1,1,1), lags = 1)
  }
  #print(team1_fit)
  
  t1xregs = data.frame(totyds = ts(t1$totyds), 
                       trnovr = ts(t1$trnovr), 
                       penyds = ts(t1$penyds),
                       ptyds  = ts(t1$ptyds))
  
  f1 = forecast(team1_fit, h = 1, xreg = t1xregs)
  ASE1 = mean((score1[l:l+1] - f1$mean) ^ 2)
  
  
  # Team 2 Model
  t2 = teams[[2]]
  score2 = t2$score
  l = length(t2$score)-1 
  score_ts2 = ts(score2[1:l])
  t2_2 = t2[1:l,]
  
  t2xregs = data.frame(totyds = ts(t2_2$totyds), 
                       trnovr = ts(t2_2$trnovr), 
                       penyds = ts(t2_2$penyds),
                       ptyds  = ts(t2_2$ptyds))
  
  if (autolag) {
    team2_fit = mlp(score_ts2, xreg = t2xregs, hd=c(8,4,2), sel.lag=TRUE, hd.auto.type = "elm",reps = 100, xreg.lags = list(1,1,1,1))
  }
  else {
    team2_fit = mlp(score_ts2, xreg = t2xregs, hd=c(8,4,2), sel.lag=TRUE, hd.auto.type = "elm",reps = 100, xreg.lags = list(1,1,1,1), lags = 1)
  }
  #print(team2_fit)
  
  t2xregs = data.frame(totyds = ts(t2$totyds), 
                       trnovr = ts(t2$trnovr), 
                       penyds = ts(t2$penyds),
                       ptyds  = ts(t2$ptyds))
  
  f2 = forecast(team2_fit, h = 1, xreg = t2xregs)
  ASE2 = mean((score2[l:l+1] - f2$mean) ^ 2)

  t1_t2_spread = plyr::round_any(as.numeric(f1$mean) - as.numeric(f2$mean),.5)
  ou = round(as.numeric(f1$mean) + as.numeric(f2$mean))
  df = data.frame(model = "MLP", team1 = team1, team1prediction=as.numeric(f1$mean) ,team2 = team2, team2prediction=as.numeric(f2$mean), line = t1_t2_spread, OU = ou, ASE_t1 = ASE1, ASE_t2 = ASE2)
  return(df)
}

# Builds Arima model for both teams.
# Models score as a predictor for team performance
# Returns the model predictions for each team, the difference,the sum and the ASE
arima_model <- function(teams, team1,team2,teamnames) 
{
  # Team 1
  t1 = teams[[1]]
  l = length(t1) - 1
  t1_2 = t1[1:l,]

  ksfit = lm(score ~ ptyds + trnovr + totyds + penyds + ptyds1 + trnovr1 + totyds1 + penyds1 , data = t1_2)
  phi = aic.wge(ksfit$residuals, p=0:7,q=0)  
  fit = arima(t1_2$score, order = c(phi$p,0,phi$q), 
              xreg = cbind(t1_2$trnovr,t1_2$ptyds,t1_2$totyds,t1_2$penyds, t1_2$trnovr1,t1_2$ptyds1,t1_2$totyds1,t1_2$penyds1))

  preds1 = predict(fit, newxreg = cbind(t1$trnovr[l:l+1],t1$ptyds[l:l+1],t1$totyds[l:l+1],t1$penyds[l:l+1],t1$trnovr1[l:l+1],t1$ptyds1[l:l+1],t1$totyds1[l:l+1],t1$penyds1[l:l+1]))
  ASE1 = mean((t1$score[l:l+1] - preds1$pred)^2)
   
  # Team 2
  t2 = teams[[2]]
  t2_2 = t2[1:l,]

  ksfit = lm(score ~ ptyds + trnovr + totyds + penyds + ptyds1 + trnovr1 + totyds1 + ptyds1 , data = t2_2)
  phi = aic.wge(ksfit$residuals, p=0:7, q=0)  
  fit = arima(t2_2$score, order = c(phi$p,0,phi$q), 
              xreg = cbind(t2_2$trnovr,t2_2$ptyds,t2_2$totyds,t2_2$penyds, t2_2$trnovr1,t2_2$ptyds1,t2_2$totyds1,t2_2$penyds1))

  preds2 = predict(fit, newxreg = cbind(t2$trnovr[l:l+1],t2$ptyds[l:l+1],t2$totyds[l:l+1],t2$penyds[l:l+1],t2$trnovr1[l:l+1],t2$ptyds1[l:l+1],t2$totyds1[l:l+1],t2$penyds1[l:l+1]))
  ASE2 = mean((t2$score[l:l+1] - preds2$pred)^2)
  
  t1_t2_spread = plyr::round_any(as.numeric(preds1$pred) - as.numeric(preds2$pred),.5)
  ou = round(as.numeric(preds1$pred) + as.numeric(preds2$pred))
  df = data.frame(model = "ARIMA",team1 = team1, team1prediction =as.numeric(preds1$pred),team2 = team2,team2prediction = as.numeric(preds2$pred), line = t1_t2_spread, OU = ou, ASE_t1 = ASE1, ASE_t2 = ASE2)
  return(df)
}  

# Builds "trend" VAR model for both teams
# Models score as a predictor for team performance
# Returns the model predictions for each team, the difference,the sum and the ASE
var_model= function(teams,team1,team2)
{
  t1 = teams[[1]]
  t2 = teams[[2]]
  l = length(t1) - 1

  t1_2 = t1[1:l,]
  t2_2 = t2[1:l,]
  
  t1var = VAR( cbind(score = t1_2$score, trnover = t1_2$trnovr, ptyds = t1_2$ptyds,totyds = t1_2$totyds, pentyds = t1_2$penyds), type = "trend")
  #print(t1var)
  preds1 = predict(t1var,n.ahead = 1)
  ASE1 = mean((t1$score[l:l+1] - preds1$fcst$score[,1])^2)
  
  t2var = VAR(cbind(score = t2_2$score, trnover = t2_2$trnovr, ptyds = t2_2$ptyds,totyds = t2_2$totyds, pentyds = t2_2$penyds), type = "trend")
  #print(t2var)
  preds2 = predict(t2var,n.ahead = 1)
  ASE2 = mean((t2$score[l:l+1] - preds2$fcst$score[,1])^2)
  
  t1_t2_spread = plyr::round_any(preds1$fcst$score[,1] - preds2$fcst$score[,1],.5)
  ou = round(preds1$fcst$score[,1] + preds2$fcst$score[,1])
  df = data.frame(model = "VAR",team1 = team1, team1prediction = preds1$fcst$score[,1], team2 = team2, team2prediction = preds2$fcst$score[,1], line = t1_t2_spread,  OU = ou, ASE_t1 = ASE1, ASE_t2 = ASE2)
  return(df)
}

ensemble_model= function(am,vm,mm,team1,team2)
{
  ensemblet1 = (am$team1prediction + vm$team1prediction + mm$team1prediction) / 3
  ensemblet2 = (am$team2prediction + vm$team2prediction + mm$team2prediction) / 3
  ensemble_line = plyr::round_any((am$line + vm$line + mm$line) / 3,.5)
  ensemblease1 = (am$ASE_t1 + vm$ASE_t1 + mm$ASE_t1)
  ensemblease2 = (am$ASE_t2 + vm$ASE_t2 + mm$ASE_t2)
  ou = round(ensemblet1 + ensemblet2)
  em = data.frame(model="Ensemble",
                  team1=team1,
                  team1prediction=ensemblet1,
                  team2=team2,
                  team2prediction=ensemblet2,
                  line = ensemble_line, 
                  OU = ou, 
                  ASE_t1 = ensemblease1, 
                  ASE_t2 = ensemblease2)
  return (em)
}

# Run all models and produce predictions for score, line, and over/under 
getPredictions = function(team1 ='DAL', team2 = 'CHI', year = 2019) 
{
  teams = getMatchupData(team1,team2,year)
  am = arima_model(teams,team1,team2)
  vm = var_model(teams,team1,team2)
  mm = mlp_model(teams,team1,team2)
  em = ensemble_model(am,vm,mm,team1,team2)
  lines = rbind(am,vm,mm,em)
  return(lines)
}

  


