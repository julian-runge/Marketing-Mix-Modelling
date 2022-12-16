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
library(Robyn)
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
View(dt_input)

## Check holidays from Prophet
# 59 countries included. If your country is not included, please manually add it.
# Tipp: any events can be added into this table, school break, events etc.
data("dt_prophet_holidays")
head(dt_prophet_holidays)

## robyn_object <- "/Users/chavi/Desktop/MMM/MMM_Meta"

### DEPRECATED: It must have extension .RDS. The object name can be different than Robyn:
robyn_object <- "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric/MyRobyn.RDS"

## Hyperparameters


InputCollect_Geo <- robyn_inputs(
  dt_input = dt_input,
  dt_holidays = dt_prophet_holidays,
  date_var = "date", # date format must be "2020-01-01"
  dep_var = "sales", # there should be only one dependent variable
  dep_var_type = "revenue", # "revenue" (ROI) or "conversion" (CPA)
  prophet_vars = c("trend", "season", "weekday", "holiday"), # "trend","season", "weekday" & "holiday"
  prophet_country = "US", # input one country. dt_prophet_holidays includes 59 countries by default
  context_vars = c("unemployment", "temperature"), # e.g. competitors, discount, unemployment etc
  paid_media_spends = c("newspaper_spend", "tv_spend"), # mandatory input
  paid_media_vars = c("newspaper_readership", "tv_gross_rating_points"), # mandatory.
  # paid_media_vars must have same order as paid_media_spends. Use media exposure metrics like
  # impressions, GRP etc. If not applicable, use spend instead.
  # organic_vars = "newspaper", # marketing activity without media spend
  # factor_vars = c("events"), # force variables in context_vars or organic_vars to be categorical,
  adstock = "geometric" # geometric, weibull_cdf or weibull_pdf.
)

print(InputCollect_Geo)

plot_adstock(plot = TRUE)
plot_saturation(plot = TRUE)

hyper_names(adstock = InputCollect_Geo$adstock, all_media = InputCollect_Geo$all_media)

hyperparameters_geo <- list(
  newspaper_spend_alphas = c(0.5, 3)
  ,newspaper_spend_gammas = c(0.3, 1)
  ,newspaper_spend_thetas = c(0.1, 0.4)
  
  ,tv_spend_alphas = c(0.5, 3)
  ,tv_spend_gammas = c(0.3, 1)
  ,tv_spend_thetas = c(0.3, 0.8)
  ,train_size = c(0.5, 0.8)
)

InputCollect_Geo <- robyn_inputs(
  dt_input = dt_input
  ,dt_holidays = dt_prophet_holidays
  ,date_var = "date"
  ,dep_var = "sales"
  ,dep_var_type = "revenue"
  ,prophet_vars = c("trend", "season", "weekday", "holiday")
  ,prophet_country = "US"
  ,context_vars = c("unemployment", "temperature")
  ,paid_media_spends = c("newspaper_spend", "tv_spend")
  ,paid_media_vars = c("newspaper_readership", "tv_gross_rating_points")
  ,window_start = "2019-07-01"
  ,window_end = "2022-07-01"
  ,adstock = "geometric"
  #   ,calibration_input = calibration_input # as in 2a-4 above
)

InputCollect_Geo <- robyn_inputs(InputCollect = InputCollect_Geo, hyperparameters = hyperparameters_geo)
print(InputCollect_Geo)

#### Check spend exposure fit if available
if (length(InputCollect_Geo$exposure_vars) > 0) {
  lapply(InputCollect_Geo$modNLS$plots, plot)
}


##### Manually save and import InputCollect as JSON file
robyn_write(InputCollect_Geo, dir = "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric")
InputCollect_Geo <- robyn_inputs(dt_input = dt_input,
                             dt_holidays = dt_prophet_holidays,
                             json_file = "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric/RobynModel-inputs.json")

## Running Model
OutputModels <- robyn_run(
  InputCollect = InputCollect_Geo, # feed in all model specification
  # cores = NULL, # default to max available
  # add_penalty_factor = FALSE, # Untested feature. Use with caution.
  iterations = 5000, 
  trials = 8, 
  ts_validation = FALSE,
  outputs = FALSE # outputs = FALSE disables direct model output - robyn_outputs()
)
print(OutputModels)

ts_validation(OutputModels)

## Check MOO (multi-objective optimization) convergence plots
OutputModels$convergence$moo_distrb_plot
OutputModels$convergence$moo_cloud_plot

## Check time-series validation plot (when ts_validation == TRUE)
# Read more and replicate results: ?ts_validation
if (OutputModels$ts_validation) OutputModels$ts_validation_plot

robyn_object2 <- "/Users/chavi/Desktop/MMM/MMM_Meta/Geometric/MyRobyn2.RDS"

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect_geo <- robyn_outputs(
  InputCollect_Geo, OutputModels,
  pareto_fronts = "auto",
  calibration_constraint = c(0.01, 0.1), # range c(0.01, 0.1) & default at 0.1
  csv_out = "pareto", # "pareto", "all", or NULL (for none)
  clusters = TRUE, # Set to TRUE to cluster similar models by ROAS. See ?robyn_clusters
  plot_pareto = TRUE, # Set to FALSE to deactivate plotting and saving model one-pagers
  plot_folder = robyn_object2, # path for plots export
  export = TRUE # this will create files locally
)
print(OutputCollect_geo)

## Calculate Pareto optimality, cluster and export results and plots. See ?robyn_outputs
OutputCollect2 <- robyn_outputs(
  InputCollect, OutputModels,
  pareto_fronts = "auto",
  calibration_constraint = c(0.01, 0.1), # range c(0.01, 0.1) & default at 0.1
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
print(OutputCollect_geo)
select_model <- "1_357_4" # Pick one of the models from OutputCollect to proceed

#### Since 3.7.1: JSON export and import (faster and lighter than RDS files)
ExportedModel <- robyn_write(InputCollect_Geo, OutputCollect_geo, select_model)
print(ExportedModel)

###### DEPRECATED (<3.7.1) (might work)
ExportedModelOld <- robyn_save(
   robyn_object = robyn_object, # model object location and name
   select_model = select_model, # selected model ID
   InputCollect = InputCollect_Geo,
   OutputCollect = OutputCollect_geo)
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
AllocatorCollect1 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_historical_response",
  channel_constr_low = 0.7,
  channel_constr_up = c(1.5, 1.5),
  export = TRUE,
  date_min = "2019-07-01",
  date_max = "2022-07-01"
)
print(AllocatorCollect1)
plot(AllocatorCollect1)

# Run the "max_response_expected_spend" scenario: "What's the maximum response for a given
# total spend based on historical saturation and what is the spend mix?" "optmSpendShareUnit"
# is the optimum spend share.
AllocatorCollect2 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7),
  channel_constr_up = c(1.5, 1.5),
  expected_spend = 300000, # Total spend to be simulated
  expected_spend_days = 9, # Duration of expected_spend in days
  export = TRUE
)
print(AllocatorCollect2)
AllocatorCollect2$dt_optimOut
plot(AllocatorCollect2)

## A csv is exported into the folder for further usage. Check schema here:
## https://github.com/facebookexperimental/Robyn/blob/main/demo/schema.R

## QA optimal response
# Pick any media variable: InputCollect$all_media
select_media <- "newspaper_spend"
# For paid_media_spends set metric_value as your optimal spend
metric_value <- AllocatorCollect1$dt_optimOut$optmSpendUnit[
  AllocatorCollect1$dt_optimOut$channels == select_media
]; metric_value
# # For paid_media_vars and organic_vars, manually pick a value
## metric_value <- 10000

if (TRUE) {
  optimal_response_allocator <- AllocatorCollect1$dt_optimOut$optmResponseUnit[
    AllocatorCollect1$dt_optimOut$channels == select_media
  ]
  optimal_response <- robyn_response(
    InputCollect = InputCollect_Geo,
    OutputCollect = OutputCollect_geo,
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

AllocatorCollect3 <- robyn_allocator(
  InputCollect = InputCollect_Geo,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  scenario = "max_response_expected_spend",
  channel_constr_low = c(0.7, 0.7),
  channel_constr_up = c(1.5, 1.5),
  expected_spend = 300000, # Total spend to be simulated
  expected_spend_days = 9 # Duration of expected_spend in days
)
print(AllocatorCollect3)
plot(AllocatorCollect3)

# Get response for 50k from result saved in robyn_object
Spend1 <- 50000
Response1 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_spend",
  metric_value = Spend1
)
Response1$response / Spend1 # ROI for search 80k
Response1$plot

#### Or you can call a JSON file directly (a bit slower)
Response1 <- robyn_response(
   json_file = json_file,
   dt_input = dt_input,
   dt_holidays = dt_prophet_holidays,
   media_metric = "",
   metric_value = Spend1
)

# Get response for +10%
Spend2 <- Spend1 * 1.1
Response2 <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_spend",
  metric_value = Spend2
)
Response2$response / Spend2 # ROI for search 81k
Response2$plot

# Marginal ROI of next 1000$ from 80k spend level for search
(Response2$response - Response1$response) / (Spend2 - Spend1)

## Example of getting paid media exposure response curves
imps <- 100000
response_imps <- robyn_response(
  InputCollect = InputCollect,
  OutputCollect = OutputCollect_geo,
  select_model = select_model,
  media_metric = "newspaper_readership",
  metric_value = imps
)
response_imps$response / imps * 1000
response_imps$plot



