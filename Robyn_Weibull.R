# Copyright (c) Meta Platforms, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#############################################################################################
####################         Facebook MMM Open Source - Robyn 3.8.1    ######################
####################                    Quick guide                   #######################
#############################################################################################

################################################################
#### Step 0: Setup environment

library(usethis)
usethis::edit_r_environ()

## Install, load, and check (latest) version.
## Install the stable version from CRAN.
## install.packages("Robyn")
## Install the dev version from GitHub
## install.packages("remotes") # Install remotes first if you haven't already
remotes::install_github("facebookexperimental/Robyn/R")
library(Robyn)

# Please, check if you have installed the latest version before running this demo. Update if not
# https://github.com/facebookexperimental/Robyn/blob/main/R/DESCRIPTION#L4
packageVersion("Robyn")

## Force multicore when using RStudio
Sys.setenv(R_PARALLELLY_FORK_ENABLE = "true")
options(parallelly.fork.enable = TRUE)

## Must install the python library Nevergrad once
## ATTENTION: The latest Python 3.10 version may cause Nevergrad installation error
## See here for more info about installing Python packages via reticulate
## https://rstudio.github.io/reticulate/articles/python_packages.html

## install.packages("reticulate") # Install reticulate first if you haven't already
library("reticulate") # Load the library

## Option 1: nevergrad installation via PIP (no additional installs)
## reticulate::install_miniconda(force = TRUE)
use_python("~/Library/r-miniconda-arm64/envs/r-reticulate/bin/python", required = TRUE)
py_install("nevergrad", pip = TRUE)
py_config() # Check your python version and configurations
## In case nevergrad still can't be installed,
# Sys.setenv(RETICULATE_PYTHON = "~/.virtualenvs/r-reticulate/bin/python")
# Reset your R session and re-install Nevergrad with option 1

## Running all the libraries
## library(Robyn)
remove.packages("data.table")
install.packages("data.table", type = "source",repos = "https://Rdatatable.gitlab.io/data.table")
library(stringr) 
library(lubridate, warn.conflicts = FALSE) 
library(foreach) 
library(future)
library(doFuture)
library(rngtools)
library(doRNG)
library(glmnet) 
library(car) 
library(StanHeaders)
library(prophet, warn.conflicts = FALSE)
library(rstan)
library(ggplot2)
library(gridExtra)
library(grid)
library(ggpubr)
library(see)
library(PerformanceAnalytics, warn.conflicts = FALSE)
library(nloptr)
library(minpack.lm)
library(rPref, warn.conflicts = FALSE)
library(reticulate)
library(rstudioapi)
library(readr)
library(remote)
library(prophet)

## Load Data
data("dt_simulated_weekly")
head(dt_simulated_weekly)

dt_input <- read_csv("data_mmm.csv")
head(dt_input)

## Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tipp: any events can be added into this table, school break, events etc.
data("dt_prophet_holidays")
head(dt_prophet_holidays)

## robyn_object_weibull <- "/Users/chavi/Desktop/MMM/MMM_Meta/weibull"

### DEPRECATED: It must have extension .RDS. The object name can be different than Robyn:
robyn_object_weibull <- "/Users/chavi/Desktop/MMM/MMM_Meta/weibull/MyRobyn.RDS"

## Checking multicolinearilty ##

#We are specifying sales as the dependent variable in the lm() function
mmm_1 <- lm(sales ~ facebook_newsfeed_spend + youtube_brand_spend + search_spend +
              youtube_performance_spend + newspaper_spend + tv_spend + facebook_newsfeed_impressions +
              youtube_brand_impressions + search_clicks + youtube_performance_impressions + 
              newspaper_readership + tv_gross_rating_points, data = dt_input)
summary(mmm_1)
plot(mmm_1)


#Check for multicollinearity using VIFs
library(mctest)
imcdiag(mmm_1, method = "VIF")

#or use jtools
library(jtools)
summ(mmm_1, vifs=TRUE)

#check for heteroscedasticity
#first, plot the model out and review the siduals vs fitted plot and the Sclae-Location plot
par(mfrow=c(2,2)) # put all 4 charts into 1 page
plot(mmm_1)

#Confirm with an objective test for heteroscedasticity using Breusch Pagan test and NCV test
library(lmtest)
lmtest::bptest(mmm_1)

library(car)
car::ncvTest(mmm_1)

#h0: the variance in the model is homoskedastic (what we want).
#Both returned p-values higher than significance level 0.05, 
#so we can't reject the null and can say that there are no major issues with heteroscedasticity.

#Create timeseries
## install.packages("forecast")
library(forecast)

#frequency is 52 to denote weekly as there are about 52 weeks in a year. 
#ts() needs a minimum of 2 periods (52 x 2 = 104 weeks), 
#our data has observations from 200 weeks so this should be sufficient
ts_sales <- ts(dt_input$sales, start = 1, frequency = 156)

#check class. should state "ts"
class(ts_sales)

#decompose to get the individual components for trends, seasonality, etc
ts_sales_comp <- decompose(ts_sales)

#plot out
plot(ts_sales_comp)

#### 2a-1: First, specify input variables

InputCollect_weibull <- robyn_inputs(
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  date_var = "date", # date format must be "2020-01-01"
  dep_var = "sales", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "holiday","weekday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "US", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("unemployment", "temperature"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c("facebook_newsfeed_spend", "youtube_brand_spend", "search_spend", "youtube_performance_spend"), # mandatory input
  paid_media_vars = c("facebook_newsfeed_impressions", "youtube_brand_impressions", "search_clicks", "youtube_performance_impressions"), # mandatory.
  # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
  # impressions, GRP etc. If not applicable, use spend instead.
  ## organic_vars = "newsletter", # marketing activity without media spend
  # factor_vars = c("events"), # force variables in context_vars or organic_vars to be categorical
  window_start = "2019-07-01",
  window_end = "2022-07-01",
  adstock = "weibull_pdf" # geometric, weibull_cdf or weibull_pdf.
)
print(InputCollect)

#### 2a-2: Second, define and add hyperparameters

## -------------------------------- NOTE v3.6.0 CHANGE !!! ---------------------------------- ##
## Default media variable for modelling has changed from paid_media_vars to paid_media_spends.
## hyperparameter names needs to be base on paid_media_spends names. Run:
hyper_names(adstock = InputCollect_weibull$adstock, all_media = InputCollect_weibull$all_media)
## to see correct hyperparameter names. Check GitHub homepage for background of change.
## Also calibration_input are required to be spend names.
## ------------------------------------------------------------------------------------------ ##

## 1. IMPORTANT: set plot = TRUE to see helper plots of hyperparameter's effect in transformation
plot_adstock(plot = TRUE)
plot_saturation(plot = TRUE)

## Hyperparameters

hyperparameters <- list(
  facebook_newsfeed_spend_alphas = c(0.5, 3)
  ,facebook_newsfeed_spend_gammas = c(0.3, 1)
  ,facebook_newsfeed_spend_shapes = c(2.0001, 10)
  ,facebook_newsfeed_spend_scales = c(0, 0.1)
  
  ,search_spend_alphas = c(0.5, 3)
  ,search_spend_gammas = c(0.3, 1)
  ,search_spend_shapes = c(2.0001, 10)
  ,search_spend_scales = c(0, 0.1)
  
  ,youtube_brand_spend_alphas = c(0.5, 3)
  ,youtube_brand_spend_gammas = c(0.3, 1)
  ,youtube_brand_spend_shapes = c(2.0001, 10)
  ,youtube_brand_spend_scales = c(0, 0.1)
  
  ,youtube_performance_spend_alphas = c(0.5, 3)
  ,youtube_performance_spend_gammas = c(0.3, 1)
  ,youtube_performance_spend_shapes = c(2.0001, 10)
  ,youtube_performance_spend_scales = c(0, 0.1)
  ## ,train_size = c(0.5, 0.8)
)

InputCollect_weibull <- robyn_inputs(InputCollect = InputCollect_weibull, hyperparameters = hyperparameters)
print(InputCollect_weibull)

##calibration_input <- data.frame(
   # channel name must in paid_media_vars
#   channel = c("facebook_newsfeed_spend", "youtube_brand_spend", "search_spend", "youtube_performance_spend"),
   # liftStartDate must be within input data range
#   liftStartDate = as.Date(c("2019-07-01", "2019-07-01", "2019-07-01", "2019-07-01")),
   # liftEndDate must be within input data range
#   liftEndDate = as.Date(c("2022-07-01", "2022-07-01", "2022-07-01", "2022-07-01")),
   # Provided value must be tested on same campaign level in model and same metric as dep_var_type
#   liftAbs = c(400000, 300000, 700000, 200),
   # Spend within experiment: should match within a 10% error your spend on date range for each channel from dt_input
#   spend = c(200000, 200000, 200000, 200000),
   # Confidence: if frequentist experiment, you may use 1 - pvalue
#   confidence = c(0.95, 0.95, 0.95, 0.95),
   # KPI measured: must match your dep_var
#   metric = c("revenue", "revenue", "revenue", "revenue"),
   # Either "immediate" or "total". For experimental inputs like Facebook Lift, "immediate" is recommended.
#   calibration_scope = c("immediate", "immediate", "immediate", "immediate")
#)

InputCollect_weibull <- robyn_inputs(InputCollect = InputCollect_weibull)

InputCollect_weibull <- robyn_inputs(
  hyperparameters = hyperparameters,
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  date_var = "date", # date format must be "2020-01-01"
  dep_var = "sales", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "weekday", "holiday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "US", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("unemployment", "temperature"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c("facebook_newsfeed_spend", "youtube_brand_spend", "search_spend", "youtube_performance_spend"), # mandatory input
  paid_media_vars = c("facebook_newsfeed_impressions", "youtube_brand_impressions", "search_clicks", "youtube_performance_impressions"), # mandatory
  window_start = "2019-07-01",
  window_end = "2022-07-01",
  # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
  # impressions, GRP etc. If not applicable, use spend instead.
  # organic_vars = "newspaper", # marketing activity without media spend
  # factor_vars = c("events"), # force variables in context_vars or organic_vars to be categorical,
  adstock = "weibull_pdf" # geometric, weibull_cdf or weibull_pdf.
)
print(InputCollect_weibull)


#### Check spend exposure fit if available
if (length(InputCollect_weibull$exposure_vars) > 0) {
  lapply(InputCollect_weibull$modNLS$plots, plot)
}

##### Manually save and import InputCollect as JSON file
robyn_write(InputCollect_weibull, dir = "/Users/chavi/Desktop/MMM/MMM_Meta/weibull")
InputCollect_weibull <- robyn_inputs(dt_input = dt_input,
                             dt_holidays = dt_prophet_holidays,
                             json_file = "/Users/chavi/Desktop/MMM/MMM_Meta/weibull/RobynModel-inputs.json")

## Running Model
OutputModels_weibull <- robyn_run(
  InputCollect = InputCollect_weibull, # feed in all model specification
  ## cores = 7, # default to max available
  # add_penalty_factor = FALSE, # Untested feature. Use with caution.
  iterations = 8000, 
  trials = 8, 
  outputs = FALSE, # outputs = FALSE disables direct model output - robyn_outputs()
  ts_validation = FALSE, # 3-way-split time series for NRMSE validation.
  add_penalty_factor = FALSE # Experimental feature. Use with caution.
)
print(OutputModels_weibull)

ts_validation(OutputModels_weibull)

## Check MOO (multi-objective optimization) convergence plots
OutputModels_weibull$convergence$moo_distrb_plot
OutputModels_weibull$convergence$moo_cloud_plot

## Check time-series validation plot (when ts_validation == TRUE)
# Read more and replicate results: ?ts_validation
if (OutputModels$ts_validation) OutputModels$ts_validation_plot

robyn_object2 <- "/Users/chavi/Desktop/MMM/MMM_Meta/weibull/MyRobyn2.RDS"

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect <- robyn_outputs(
  InputCollect_weibull, OutputModels_weibull,
  pareto_fronts = "auto",
  calibration_constraint = c(0.01, 0.1), # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_object2, # path for plots export
  export = TRUE # this will create files locally
)
print(OutputCollect)

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect2 <- robyn_outputs(
  InputCollect, OutputModels_weibull,
  pareto_fronts = "auto",
  calibration_constraint = 0.1, # range c(0.01, 0.1) & default at 0.1
  csv_out = "NULL", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_object2, # path for plots export
  export = TRUE # this will create files locally
)
print(OutputCollect2)

################################################################
#### Step 4: Select and save the any model

## Compare all model one-pagers and select one that mostly reflects your business reality
print(OutputCollect)
select_model <- "2_451_7" # Pick one of the models from OutputCollect to proceed

#### Since 3.7.1: JSON export and import (faster and lighter than RDS files)
ExportedModel <- robyn_write(InputCollect_weibull, OutputCollect, select_model)
print(ExportedModel)

###### DEPRECATED (<3.7.1) (might work)
ExportedModelOld <- robyn_save(
  robyn_object = robyn_object, # model object location and name
  select_model = select_model, # selected model ID
  InputCollect = InputCollect,
  OutputCollect = OutputCollect)
print(ExportedModelOld)
plot(ExportedModelOld)

################################################################
#### Step 5: Get budget allocation based on the selected model above

## Budget allocation result requires further validation. Please use this recommendation with caution.
## Don't interpret budget allocation result if selected model above doesn't meet business expectation.

# Check media summary for selected model
print(ExportedModel)

# Run ?robyn_allocator to check parameter definition
# Run the "max_historical_response" scenario: "What's the revenue lift potential with the
# same historical spend level and what is the spend mix?"
AllocatorCollect_weibull <- robyn_allocator(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_historical_response",
  channel_constr_low = c(0.7, 0.7, 0.7, 0.7),
  channel_constr_up = c(1.2, 1.5, 1.5, 1.5),
  export = TRUE,
  date_min = "2019-07-01",
  date_max = "2021-07-01"
)
print(AllocatorCollect_weibull)
plot(AllocatorCollect_weibull)

# Run the "max_response_expected_spend" scenario: "What's the maximum response for a given
# total spend based on historical saturation and what is the spend mix?" "optmSpendShareUnit"
# is the optimum spend share.
AllocatorCollect_we2 <- robyn_allocator(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7, 0.7, 0.7),
  channel_constr_up = c(1.2, 1.5, 1.5, 1.5),
  expected_spend = 700000, # Total spend to be simulated
  expected_spend_days = 8, # Duration of expected_spend in days
  export = TRUE
)
print(AllocatorCollect_we2)
AllocatorCollect_we2$dt_optimOut
plot(AllocatorCollect_we2)

## A csv is exported into the folder for further usage. Check schema here:
## https://github.com/facebookexperimental/Robyn/blob/main/demo/schema.R

## QA optimal response
# Pick any media variable: InputCollect$all_media
select_media <- "youtube_performance_spend"
# For paid_media_spends set metric_value as your optimal spend
metric_value <- AllocatorCollect_weibull$dt_optimOut$optmSpendUnit[
  AllocatorCollect_weibull$dt_optimOut$channels == select_media
]; metric_value
# # For paid_media_vars and organic_vars, manually pick a value
metric_value <- 350000

if (TRUE) {
  optimal_response_allocator <- AllocatorCollect1$dt_optimOut$optmResponseUnit[
    AllocatorCollect1$dt_optimOut$channels == select_media
  ]
  optimal_response <- robyn_response(
    InputCollect = InputCollect_weibull,
    OutputCollect = OutputCollect,
    select_model = select_model,
    select_build = 0,
    media_metric = select_media,
    metric_value = metric_value
  )
  plot(optimal_response$plot)
  if (length(optimal_response_allocator) > 0) {
    cat("QA if results from robyn_allocator and robyn_response agree: ")
    cat(round(optimal_response_allocator) == round(optimal_response$response), "( ")
    cat(optimal_response$response, "==", optimal_response_allocator, ")\n")
  }
}

################################################################
#### Step 6: Model refresh based on selected model and saved results "Alpha" [v3.7.1]

## Must run robyn_write() (manually or automatically) to export any model first, before refreshing.
## The robyn_refresh() function is suitable for updating within "reasonable periods".
## Two situations are considered better to rebuild model:
## 1. most data is new. If initial model has 100 weeks and 80 weeks new data is added in refresh,
## it might be better to rebuild the model. Rule of thumb: 50% of data or less can be new.
## 2. new variables are added.

##### Manually save and import InputCollect as JSON file

RobynRefresh <- robyn_inputs(
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 10,
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 5,
  json_file = "/Users/chavi/Desktop/MMM/MMM_Meta/weibull/RobynModel-inputs.json")

robyn_write(RobynRefresh, dir = "/Users/chavi/Desktop/MMM/MMM_Meta/weibull")

# Provide JSON file with your InputCollect and ExportedModel specifications
# It can be any model, initial or a refresh model
json_file <- "~/Desktop/MMM/MMM_Meta/weibull/Robyn_202212081303_init/RobynModel-2_451_7.json"
RobynRefresh <- robyn_refresh(
  json_file = json_file,
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 5,
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 3
)

json_file_rf1 <- "~/Desktop/Robyn_202208231837_init/Robyn_202208231841_rf1/RobynModel-1_12_5.json"
RobynRefresh <- robyn_refresh(
  json_file = json_file_rf1,
  dt_input = dt_,
  dt_holidays = dt_prophet_holidays,
  refresh_steps = 5,
  refresh_iters = 1000, # 1k is an estimation
  refresh_trials = 1
)

InputCollect_weibull <- RobynRefresh$listRefresh1$InputCollect
OutputCollect <- RobynRefresh$listRefresh1$OutputCollect
select_model <- RobynRefresh$listRefresh1$OutputCollect$selectID

###### DEPRECATED (<3.7.1) (might work)
# # Run ?robyn_refresh to check parameter definition
# Robyn <- robyn_refresh(
#   robyn_object = robyn_object,
#   dt_input = dt_simulated_weekly,
#   dt_holidays = dt_prophet_holidays,
#   refresh_steps = 4,
#   refresh_mode = "manual",
#   refresh_iters = 1000, # 1k is estimation. Use refresh_mode = "manual" to try out.
#   refresh_trials = 1
# )

## Besides plots: there are 4 CSV outputs saved in the folder for further usage
# report_hyperparameters.csv, hyperparameters of all selected model for reporting
# report_aggregated.csv, aggregated decomposition per independent variable
# report_media_transform_matrix.csv, all media transformation vectors
# report_alldecomp_matrix.csv,all decomposition vectors of independent variables

AllocatorCollect <- robyn_allocator(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7, 0.7, 0.7),
  channel_constr_up = c(1.2, 1.5, 1.5, 1.5),
  expected_spend = 700000, # Total spend to be simulated
  expected_spend_days = 9 # Duration of expected_spend in days
)
print(AllocatorCollect)
plot(AllocatorCollect)

# Get response for 50k from result saved in robyn_object
Spend1 <- 700000
Response1 <- robyn_response(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "youtube_performance_spend",
  metric_value = Spend1
)
Response1$response / Spend1 # ROI for search 80k
Response1$plot

#### Or you can call a JSON file directly (a bit slower)
Response1 <- robyn_response(
  json_file = json_file,
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  media_metric = "facebook_newsfeed_spend",
  metric_value = Spend1
)
Response1$plot

# Get response for +10%
Spend2 <- Spend1 * 1.1
Response2 <- robyn_response(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "youtube_brand_spend",
  metric_value = Spend2
)
Response2$response / Spend2 # ROI for search 81k
Response2$plot

# Marginal ROI of next 1000$ from 80k spend level for search
(Response2$response - Response1$response) / (Spend2 - Spend1)

## Example of getting paid media exposure response curves
imps <- 500000
response_imps <- robyn_response(
  InputCollect = InputCollect_weibull,
  OutputCollect = OutputCollect,
  select_model = select_model,
  media_metric = "youtube_performance_impressions",
  metric_value = imps
)
response_imps$response / imps * 1000
response_imps$plot

library(knitr)
library(markdown)
knitr::knit2html("weibull_weekly.R")
knitr::knit2html("Robyn.R")



