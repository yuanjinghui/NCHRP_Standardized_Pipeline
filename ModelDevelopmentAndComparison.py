"""
Created on Mon Feburary 11 20:01:56 2021

@author: Jinghui Yuan @ UCF
"""
import pandas as pd
import numpy as np
import os
import statsmodels.formula.api as smf
from statsmodels.genmod import bayes_mixed_glm
import statsmodels.api as sm
import random
from sklearn import metrics
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pymc3 as pm
from pymc3 import *
import matplotlib.pyplot as plt
import re
import arviz as az
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVR
import theano.tensor as tt
from sklearn import linear_model
from fastprogress.fastprogress import force_console_behavior
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.neural_network import MLPRegressor
from time import sleep
from patsy import dmatrices


def data_transformation(data, number_of_years):
    """

    :param data:
    :param number_of_years:
    :return:
    """
    data['RuralUrban'] = np.where(data['Urban_Code'] <= 99998, 1, 0)

    # delete the segments shorter than 0.1 mile
    data = data[data['Miles'] >= 0.1].reset_index(drop=True)

    # average between up and down stream detectors
    data['AvgVolume'] = data.loc[:, ['VolumeUp', 'VolumeDown']].mean(axis=1)
    data['AvgSpeed'] = data.loc[:, data.columns.str.contains('AvgSpeed')].mean(axis=1)
    data['StdSpeed'] = data.loc[:, data.columns.str.contains('StdSpeed')].mean(axis=1)
    data['CoefOfVarSpeed'] = data.loc[:, data.columns.str.contains('CoefOfVarSpeed')].mean(axis=1)
    data['AvgOccupancy'] = data.loc[:, data.columns.str.contains('AvgOccupancy')].mean(axis=1)
    data['StdOccupancy'] = data.loc[:, data.columns.str.contains('StdOccupancy')].mean(axis=1)
    data['CoefOfVarOccupancy'] = data.loc[:, data.columns.str.contains('CoefOfVarOccupancy')].mean(axis=1)

    # nature log of volume and segment length
    data['LogVolume'] = np.log(data['AvgVolume'])
    data['LogSegLength'] = np.log(data['Miles'])
    # data['log_AADT'] = np.log(data['AADT'])

    # speed limit
    data = data.sort_values(by=['RouteId', 'SegmentId']).reset_index(drop=True)
    data['SpeedLimit'] = data['SpeedLimit'].fillna(method='ffill', axis=0)

    # create categorical variable for ThruLanes (0 (<=4), 1 (5-7), 2 (>=8))
    data['LaneNumber'] = 0
    data.loc[(data['ThruLanes'] >= 5) & (data['ThruLanes'] <= 7), 'LaneNumber'] = 1
    data.loc[data['ThruLanes'] >= 8, 'LaneNumber'] = 2

    # get annual average crashes (divide by 2 years)
    data['TotalCrashes'] = data['TotalCrashes']/number_of_years
    data['KABCCrashes'] = data['KABCCrashes']/number_of_years
    data['OCrashes'] = data['OCrashes']/number_of_years

    return data


# descriptive statistics
def get_descriptive_statistics(data, path, variable_list):
    """

    :param data:
    :param path:
    :param variable_list:
    :return:
    """
    weekday_hourly_model_data_statistics = pd.DataFrame(data[variable_list].describe())
    weekday_hourly_model_data_statistics.transpose().to_csv(path)
    return weekday_hourly_model_data_statistics


def get_train_test(train_segments, test_segments, data):
    """

    :param train_segments:
    :param test_segments:
    :param data:
    :return:
    """
    train_data = data[data['SegmentId'].isin(train_segments)].reset_index(drop=True)
    test_data = data[data['SegmentId'].isin(test_segments)].reset_index(drop=True)
    return train_data, test_data


def data_transform_descriptive_split(integrated_model_data_path, rename_dic, number_of_years):
    """

    :param integrated_model_data_path:
    :param rename_dic:
    :param number_of_years:
    :return:
    """
    for aggregation_level in list(['weekday_hourly', 'weekend_hourly', 'weekday_peak', 'day_of_week', 'average_daily']):
        traffic_crash_geometric = pd.read_csv(os.path.join(integrated_model_data_path, '{}_model_data.csv'.format(aggregation_level)), index_col=0)

        traffic_crash_geometric = traffic_crash_geometric.rename(columns=dict(zip(rename_dic["OriginalName"], rename_dic["StandardName"])))
        traffic_crash_geometric = data_transformation(traffic_crash_geometric, number_of_years)

        model_data_statistics = get_descriptive_statistics(traffic_crash_geometric, os.path.join(integrated_model_data_path, '{}_model_data_statistics.csv'.format(aggregation_level)), variable_list)

        # split the train and test segment ids
        if aggregation_level == 'weekday_hourly':
            # split train and test segments
            total_segments = traffic_crash_geometric['SegmentId'].unique().tolist()
            len(total_segments)
            n_segments_train = int(len(total_segments) * 0.7)
            random.seed(0)
            train_segments = random.sample(total_segments, n_segments_train)
            test_segments = [i for i in total_segments if i not in train_segments]

        model_data_train, model_data_test = get_train_test(train_segments, test_segments, traffic_crash_geometric)
        model_data_train.to_csv(os.path.join(integrated_model_data_path, '{}_model_data_train.csv'.format(aggregation_level)))
        model_data_test.to_csv(os.path.join(integrated_model_data_path, '{}_model_data_test.csv'.format(aggregation_level)))

        print(aggregation_level)


# relative path for the rename dictionary
rename_dict_path = 'rename_dictionary.csv'

# read the rename dictionary
rename_dic = pd.read_csv(rename_dict_path)

# specify the number of years
number_of_years = 2

# relative path for the output integrated model data
integrated_model_data_path = 'Model'

# variable list for descriptive statistics
variable_list = ['SegmentId', 'TotalCrashes', 'Miles', 'LaneNumber', 'RuralUrban', 'SpeedLimit', 'IRI', 'AvgVolume', 'AvgSpeed',
                 'StdSpeed', 'CoefOfVarSpeed', 'AvgOccupancy', 'StdOccupancy', 'CoefOfVarOccupancy']

# call the function to get data transformation, descriptive statistics, and train/test split
data_transform_descriptive_split(integrated_model_data_path, rename_dic, number_of_years)


############################################################
# Model Development #############################
############################################################
corrMatrix = weekday_hourly_model_data[['log_seg_length', 'lane_number', 'rural_urban', 'speed_limit', 'IRI', 'log_volume', 'avg_speed', 'std_speed', 'avg_occupancy', 'std_occupancy']].corr()

formula = "total_crashes ~ log_seg_length + C(lane_number) + C(rural_urban) + C(speed_limit) + IRI + log_volume + avg_speed + std_speed"
model = smf.glm(formula=formula, data=weekday_hourly_model_data_train, family=sm.families.NegativeBinomial()).fit()

formula = "total_crashes ~ log_seg_length + IRI + log_volume + avg_speed"
model = smf.glm(formula=formula, data=weekday_hourly_model_data_train, family=sm.families.NegativeBinomial()).fit()

formula = "total_crashes ~ log_seg_length + log_volume + avg_speed"
model = smf.glm(formula=formula, data=weekday_hourly_model_data_train, family=sm.families.NegativeBinomial()).fit()
model.summary()
model.aic
model.bic

y_pred = model.predict(weekday_hourly_model_data_test)
y_true = weekday_hourly_model_data_test['total_crashes'].tolist()
weekday_hourly_model_data_test['y_pred'] = y_pred
mae = metrics.mean_absolute_error(y_true, y_pred)
rmse = (np.sqrt(metrics.mean_squared_error(y_true, y_pred)))

# calculate log-likelihood on test data
model_new = smf.glm(formula=formula, data=weekday_hourly_model_data_test, family=sm.families.NegativeBinomial())
log_like = model_new.loglike(model.params)

# weekday period crash frequency
weekday_hourly_model_data_test['period'] = 0  # night time
weekday_hourly_model_data_test.loc[weekday_hourly_model_data_test['hour'].isin([6, 7, 8]), 'period'] = 1  # AM peak
weekday_hourly_model_data_test.loc[weekday_hourly_model_data_test['hour'].isin([16, 17, 18]), 'period'] = 2  # PM peak
weekday_hourly_model_data_test.loc[weekday_hourly_model_data_test['hour'].isin([9, 10, 11, 12, 13, 14, 15]), 'period'] = 3  # Daytime Off-peak

evaluate_weekday_hourly2weekday_period = weekday_hourly_model_data_test.groupby(by=['zone_id_st', 'period'], as_index=False).agg({'total_crashes': 'sum', 'y_pred': 'sum'})
mae = metrics.mean_absolute_error(evaluate_weekday_hourly2weekday_period['total_crashes'], evaluate_weekday_hourly2weekday_period['y_pred'])
rmse = (np.sqrt(metrics.mean_squared_error(evaluate_weekday_hourly2weekday_period['total_crashes'], evaluate_weekday_hourly2weekday_period['y_pred'])))

# annual weekday crash frequency
evaluate_weekday_hourly2weekday_annual = weekday_hourly_model_data_test.groupby(by=['zone_id_st'], as_index=False).agg({'total_crashes': 'sum', 'y_pred': 'sum'})
mae = metrics.mean_absolute_error(evaluate_weekday_hourly2weekday_annual['total_crashes'], evaluate_weekday_hourly2weekday_annual['y_pred'])
rmse = (np.sqrt(metrics.mean_squared_error(evaluate_weekday_hourly2weekday_annual['total_crashes'], evaluate_weekday_hourly2weekday_annual['y_pred'])))

# annual crash frequency (weekday_hourly_model + weekend_hourly_model)
evaluate_weekday_hourly2annual_1 = pd.concat([evaluate_weekday_hourly2weekday_annual[['zone_id_st', 'total_crashes', 'y_pred']],
                                             evaluate_weekend_hourly2weekend_annual[['zone_id_st', 'total_crashes', 'y_pred']]], axis=0)
evaluate_weekday_hourly2annual_1 = evaluate_weekday_hourly2annual_1.groupby(by=['zone_id_st'], as_index=False).agg({'total_crashes': 'sum', 'y_pred': 'sum'})
mae = metrics.mean_absolute_error(evaluate_weekday_hourly2annual_1['total_crashes'], evaluate_weekday_hourly2annual_1['y_pred'])
rmse = (np.sqrt(metrics.mean_squared_error(evaluate_weekday_hourly2annual_1['total_crashes'], evaluate_weekday_hourly2annual_1['y_pred'])))

# annual crash frequency (weekday_hourly_model + day_of_week_model)
evaluate_weekday_hourly2annual_2 = pd.concat([evaluate_weekday_hourly2weekday_annual[['zone_id_st', 'total_crashes', 'y_pred']],
                                              evaluate_day_of_week2weekend_annual[['zone_id_st', 'total_crashes', 'y_pred']]], axis=0)
evaluate_weekday_hourly2annual_2 = evaluate_weekday_hourly2annual_2.groupby(by=['zone_id_st'], as_index=False).agg({'total_crashes': 'sum', 'y_pred': 'sum'})
mae = metrics.mean_absolute_error(evaluate_weekday_hourly2annual_2['total_crashes'], evaluate_weekday_hourly2annual_2['y_pred'])
rmse = (np.sqrt(metrics.mean_squared_error(evaluate_weekday_hourly2annual_2['total_crashes'], evaluate_weekday_hourly2annual_2['y_pred'])))


# Calculating VIF
def calc_vif(X):
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif = vif.sort_values(by='VIF').reset_index(drop=True)

    return vif


def calc_vif_automatic(X):
    vif = calc_vif(X)
    num_var = len(vif)
    for i in range(num_var):
        if vif.iloc[-1, 1] > 10:
            new_var_list = vif[:-1]['variables'].tolist()
            X = X[new_var_list]
            vif = calc_vif(X)

            print(i, vif.iloc[-1, 1])
        else:
            return vif, new_var_list


############################################################
# Basic Negative Binomial Model  ##########################
############################################################
# vif = calc_vif(Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number', 'rural_urban', 'IRI', 'log_volume', 'std_speed', 'avg_occupancy']])

# vif, new_var_list = calc_vif_automatic(Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number', 'state', 'rural_urban', 'speed_limit', 'IRI', 'log_volume', 'avg_speed', 'std_speed', 'avg_occupancy', 'std_occupancy']])
# corrMatrix = Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number', 'state', 'rural_urban', 'speed_limit', 'IRI', 'log_volume', 'avg_speed', 'std_speed', 'avg_occupancy', 'std_occupancy']].corr()
vif, new_var_list = calc_vif_automatic(Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number_1', 'lane_number_2', 'rural_urban', 'speed_limit', 'IRI', 'log_volume', 'avg_speed', 'std_speed', 'avg_occupancy', 'std_occupancy']])
vif = calc_vif(Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number_1', 'lane_number_2', 'rural_urban', 'log_volume', 'std_speed']])
corrMatrix = Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number_1', 'lane_number_2', 'state', 'rural_urban', 'speed_limit', 'IRI', 'log_volume', 'avg_speed', 'std_speed', 'avg_occupancy', 'std_occupancy']].corr()

# formula = "total_crashes ~ log_seg_length + C(lane_number) + rural_urban + IRI + log_volume + std_speed + avg_occupancy"
formula = "total_crashes ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
model = smf.glm(formula=formula, data=Combined_weekday_peak_model_data_train, family=sm.families.NegativeBinomial()).fit()
model.summary()
model.aic
model.bic


formula = "total_crashes ~ lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
model = smf.glm(formula=formula, data=Combined_weekday_peak_model_data_train, offset=Combined_weekday_peak_model_data_train['log_seg_length'], family=sm.families.NegativeBinomial()).fit()
model.summary()
model.aic
model.bic


def evaluation_metrics(test, predictions):
    rmse = (np.sqrt(metrics.mean_squared_error(test, predictions)))

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mae


rmse, mae = evaluation_metrics(Combined_weekday_peak_model_data_test['total_crashes'].values, model.predict(Combined_weekday_peak_model_data_test, offset=Combined_weekday_peak_model_data_test['log_seg_length']))

############################################################
# Basic Poisson Model  ##########################
############################################################

formula = "total_crashes ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
model = smf.glm(formula=formula, data=Combined_weekday_peak_model_data_train, family=sm.families.Poisson()).fit()
model.summary()
model.aic
model.bic

rmse, mae = evaluation_metrics(Combined_weekday_peak_model_data_test['total_crashes'].values, model.predict(Combined_weekday_peak_model_data_test))


formula = "total_crashes ~ lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
model = smf.glm(formula=formula, data=Combined_weekday_peak_model_data_train, offset=Combined_weekday_peak_model_data_train['log_seg_length'], family=sm.families.Poisson()).fit()
model.summary()
model.aic
model.bic

rmse, mae = evaluation_metrics(Combined_weekday_peak_model_data_test['total_crashes'].values, model.predict(Combined_weekday_peak_model_data_test, offset=Combined_weekday_peak_model_data_test['log_seg_length']))

############################################################
# Basic Poisson Lognormal Model (I did not find a good way of developing Poisson Lognormal model in python, and I did this model in R) ##########################
############################################################

# # Set the response, predictor, and random columns
# y = "total_crashes"
# x = ["log_seg_length","lane_number_1","lane_number_2","rural_urban","log_volume","std_speed"]
# z = 0
#
# # Train and view the model
# h2o_glm = H2OGeneralizedLinearEstimator(HGLM=True,
#                                         family="gaussian",
#                                         rand_family=["gaussian"],
#                                         random_columns=[z],
#                                         rand_link=["identity"],
#                                         calc_like=True)
# h2o_glm.train(x=x, y=y, training_frame=h2o.H2OFrame(Combined_weekday_peak_model_data_train))
# print(h2o_glm)

############################################################
# Basic Zero-Inflated Poisson Model  ##########################
############################################################

formula = "total_crashes ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
y_train, X_train = dmatrices(formula, Combined_weekday_peak_model_data_train, return_type='dataframe')
y_test, X_test = dmatrices(formula, Combined_weekday_peak_model_data_test, return_type='dataframe')

model = sm.ZeroInflatedPoisson(endog=y_train, exog=X_train, exog_infl=X_train, inflation='logit').fit_regularized(maxiter = 100)
model.summary()
model.aic
model.bic

rmse, mae = evaluation_metrics(Combined_weekday_peak_model_data_test['total_crashes'].values, model.predict(X_test, exog_infl=X_test))


############################################################
# Basic Zero-Inflated Negative Binomial Model  ##########################
############################################################

formula = "total_crashes ~ log_seg_length + lane_number_1 + lane_number_2 + rural_urban + log_volume + std_speed"
y_train, X_train = dmatrices(formula, Combined_weekday_peak_model_data_train, return_type='dataframe')
y_test, X_test = dmatrices(formula, Combined_weekday_peak_model_data_test, return_type='dataframe')

model = sm.ZeroInflatedNegativeBinomialP(endog=y_train, exog=X_train, exog_infl=X_train, inflation='logit').fit_regularized(maxiter = 100)
model = sm.ZeroInflatedNegativeBinomialP(y_train, X_train).fit_regularized(maxiter = 100)
model.summary()
model.aic
model.bic

zinb_pred = model.predict(X_test, exog_infl=np.ones((len(X_test), 1)))

rmse, mae = evaluation_metrics(Combined_weekday_peak_model_data_test['total_crashes'].values, zinb_pred)


############################################################
# Bayesian Negative Binomial Model  ##########################
############################################################
# offset segment length
def model_factory(data):
    with pm.Model() as model:
        # define priors
        alpha = pm.Gamma('alpha', alpha=0.001, beta=0.001)
        beta = pm.Normal('beta', 0, 100, shape=6)
        lam = beta[0] + \
              data['log_seg_length'] + \
              beta[1] * data['lane_number_1'] + \
              beta[2] * data['lane_number_2'] + \
              beta[3] * data['rural_urban'] + \
              beta[4] * data['log_volume'] + \
              beta[5] * data['std_speed']

        # Define likelihood
        y = pm.NegativeBinomial('y', mu=np.exp(lam), alpha=alpha, observed=data['total_crashes'])

    return model


# train the model
with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(2000, tune=2000, chains=3, cores=3)


az.summary(trace)
pm.gelman_rubin(trace)
# test the model performance
with model_factory(Combined_weekday_peak_model_data_test):
    ppc = pm.sample_posterior_predictive(trace) #or whatever

y=np.asarray(ppc['y']).mean(axis=0)
y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
# Combined_weekday_peak_model_data_test['y_pred'] = y
# mae = metrics.mean_absolute_error(y_true, y)
# rmse = (np.sqrt(metrics.mean_squared_error(y_true, y)))
rmse, mae = evaluation_metrics(y_true, y)


############################################################
# Bayesian Poisson Lognormal Model  ##########################
############################################################
def model_factory(data):
    with pm.Model() as model:
        # define priors
        beta = pm.Normal('beta', 0, 100, shape=7)
        sigma_a = pm.Gamma('sigma_a', alpha=1, beta=1)
        a = pm.Normal('a', mu=0.01, sigma=sigma_a, shape=n_obs)
        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed'] + a[idx]

        # Define likelihood
        y = pm.Poisson('y', mu=np.exp(lam), observed=data['total_crashes'])

    return model


# pm.model_to_graphviz(model_factory)
# train the model
with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(2000, tune=2000, chains=3, cores=3)

az.summary(trace, var_names=["beta", 'sigma_a'])
pm.gelman_rubin(trace)
pm.traceplot(trace)


# test the model performance
def model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta"]):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]
    trace_df_summary.loc['beta[0]', 'mean']

    lam = np.exp(trace_df_summary.loc['beta[0]', 'mean'] + \
          trace_df_summary.loc['beta[1]', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['beta[2]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['beta[3]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['beta[4]', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['beta[5]', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['beta[6]', 'mean'] * Combined_weekday_peak_model_data_test['std_speed']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta"])


############################################################
# Bayesian Random Effect Negative Binomial Model  ##########
############################################################
idx_segment = pd.Index(Combined_weekday_peak_model_data_train["Seg_Id"].unique()).get_indexer(Combined_weekday_peak_model_data_train.Seg_Id)
n_segments = len(Combined_weekday_peak_model_data_train["Seg_Id"].unique())

idx_roadway = pd.Index(Combined_weekday_peak_model_data_train["RoadNumber"].unique()).get_indexer(Combined_weekday_peak_model_data_train.RoadNumber)
n_roadway = len(Combined_weekday_peak_model_data_train["RoadNumber"].unique())


def model_factory(data):
    with pm.Model() as model:
        # define priors
        alpha = pm.Gamma('alpha', alpha=0.001, beta=0.001)
        beta = pm.Normal('beta', 0, 100, shape=7)
        mu_a = pm.Normal('mu_a', mu=0., sigma=100)
        sigma_a = pm.HalfNormal('sigma_a', 5.)
        a = pm.Normal('a', mu=mu_a, sigma=sigma_a, shape=n_roadway)
        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed'] + a[idx_roadway]

        # Define likelihood
        y = pm.NegativeBinomial('y', mu=np.exp(lam), alpha=alpha, observed=data['total_crashes'])

    return model


# train the model
with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(10000, tune=2000, chains=3, cores=3)


az.summary(trace, var_names=["beta", 'alpha', 'mu_a', 'sigma_a'])
pm.gelman_rubin(trace)
pm.traceplot(trace)


def model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta", 'mu_a']):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]

    lam = np.exp(trace_df_summary.loc['beta[0]', 'mean'] + \
          trace_df_summary.loc['beta[1]', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['beta[2]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['beta[3]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['beta[4]', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['beta[5]', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['beta[6]', 'mean'] * Combined_weekday_peak_model_data_test['std_speed'] +
                 trace_df_summary.loc['mu_a', 'mean']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta", 'mu_a'])


############################################################
# Bayesian Random Effect Poisson Lognormal Model  ##########################
############################################################
idx = np.array(Combined_weekday_peak_model_data_train.index)
n_obs = len(Combined_weekday_peak_model_data_train)


# define model (manually specified)
def model_factory(data):
    with pm.Model() as model:
        # define priors
        beta = pm.Normal('beta', 0, 100, shape=7)
        sigma_a = pm.Gamma('sigma_a', alpha=1, beta=1)
        a = pm.Normal('a', mu=0.01, sigma=sigma_a, shape=n_obs)

        mu_b = pm.Normal('mu_b', mu=0., sigma=100)
        sigma_b = pm.HalfNormal('sigma_b', 5.)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_roadway)

        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed'] + a[idx] + b[idx_roadway]

        # Define likelihood
        y = pm.Poisson('y', mu=np.exp(lam), observed=data['total_crashes'])

    return model


with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(5000, tune=5000, chains=3, cores=3)

az.summary(trace, var_names=["beta", 'sigma_a', 'mu_b', 'sigma_b'])
pm.gelman_rubin(trace)
# pm.traceplot(trace)
# plt.show()


def model_eval(trace, Combined_weekday_peak_model_data_test, vars):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]

    lam = np.exp(trace_df_summary.loc['beta[0]', 'mean'] + \
          trace_df_summary.loc['beta[1]', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['beta[2]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['beta[3]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['beta[4]', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['beta[5]', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['beta[6]', 'mean'] * Combined_weekday_peak_model_data_test['std_speed'] +
                 trace_df_summary.loc['mu_b', 'mean']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta", 'mu_b'])


############################################################
# Bayesian Zero-inflated Negative binomial  ##########################
############################################################
n_exp = len(Combined_weekday_peak_model_data_train)


def model_factory(data):
    with pm.Model() as model:
        psi = pm.Uniform('psi')
        beta = pm.Normal('beta', 0, 100, shape=7)
        alpha = pm.Gamma('alpha', alpha=np.std(data['total_crashes']), beta=5)

        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed']

        nb = pm.ZeroInflatedNegativeBinomial('nb', psi=psi, mu=np.exp(lam), alpha=alpha,
                                             observed=data['total_crashes'])

    return model


with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(5000, tune=5000, chains=3, cores=3)

az.summary(trace, var_names=["beta", 'psi', 'alpha'])
pm.gelman_rubin(trace)


def model_eval(trace, Combined_weekday_peak_model_data_test, vars):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]

    lam = np.exp(trace_df_summary.loc['beta[0]', 'mean'] + \
          trace_df_summary.loc['beta[1]', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['beta[2]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['beta[3]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['beta[4]', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['beta[5]', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['beta[6]', 'mean'] * Combined_weekday_peak_model_data_test['std_speed']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta"])


############################################################
# Bayesian Zero-inflated Poisson Lognormal  ##########################
############################################################
n_obs = len(Combined_weekday_peak_model_data_train)
idx = np.array(Combined_weekday_peak_model_data_train.index)


# define model (manually specified)
def model_factory(data):
    with pm.Model() as model:
        psi = pm.Uniform('psi')
        beta = pm.Normal('beta', 0, 100, shape=7)
        sigma_a = pm.Gamma('sigma_a', alpha=1, beta=1)
        a = pm.Normal('a', mu=0, sigma=sigma_a, shape=n_obs)

        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed'] + a[idx]

        nb = pm.ZeroInflatedPoisson('nb', psi=psi, theta=np.exp(lam), observed=data['total_crashes'])

    return model


with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(2000, tune=2000, chains=3, cores=3)

az.summary(trace, var_names=["beta", 'psi', 'sigma_a'])
pm.gelman_rubin(trace)

mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta"])


############################################################
# Bayesian Random Effect Zero-inflated Poisson Lognormal  ####
##################f##########################################
n_obs = len(Combined_weekday_peak_model_data_train)
idx = np.array(Combined_weekday_peak_model_data_train.index)


# define model (manually specified)
def model_factory(data):
    with pm.Model() as model:
        psi = pm.Uniform('psi')
        beta = pm.Normal('beta', 0, 100, shape=7)
        sigma_a = pm.Gamma('sigma_a', alpha=1, beta=1)
        a = pm.Normal('a', mu=0.001, sigma=sigma_a, shape=n_obs)

        mu_b = pm.Normal('mu_b', mu=0., sigma=100)
        sigma_b = pm.HalfNormal('sigma_b', 5.)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_roadway)

        lam = beta[0] + \
              beta[1] * data['log_seg_length'] + \
              beta[2] * data['lane_number_1'] + \
              beta[3] * data['lane_number_2'] + \
              beta[4] * data['rural_urban'] + \
              beta[5] * data['log_volume'] + \
              beta[6] * data['std_speed'] + a[idx] + b[idx_roadway]

        nb = pm.ZeroInflatedPoisson('nb', psi=psi, theta=np.exp(lam), observed=data['total_crashes'])

    return model


with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(10000, tune=5000, chains=3, cores=3)

az.summary(trace, var_names=["beta", 'psi', 'sigma_a', 'mu_b', 'sigma_b'])
pm.gelman_rubin(trace)


def model_eval(trace, Combined_weekday_peak_model_data_test, vars):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]

    lam = np.exp(trace_df_summary.loc['beta[0]', 'mean'] + \
          trace_df_summary.loc['beta[1]', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['beta[2]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['beta[3]', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['beta[4]', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['beta[5]', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['beta[6]', 'mean'] * Combined_weekday_peak_model_data_test['std_speed'] +
                 trace_df_summary.loc['mu_b', 'mean']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta", 'mu_b'])


############################################################
# Bayesian Random Parameters Poisson Lognormal  ####
# #################f##########################################
idx = np.array(Combined_weekday_peak_model_data_train.index)
n_obs = len(Combined_weekday_peak_model_data_train)

idx_roadway = pd.Index(Combined_weekday_peak_model_data_train["RoadNumber"].unique()).get_indexer(Combined_weekday_peak_model_data_train.RoadNumber)
n_roadway = len(Combined_weekday_peak_model_data_train["RoadNumber"].unique())


# define model (manually specified)
def model_factory(data):
    with pm.Model() as model:

        # define priors
        beta0 = pm.Normal('beta0', mu=0, sigma=100)
        b1 = pm.Normal('b1', mu=0, sigma=100)
        b2 = pm.Normal('b2', mu=0, sigma=100)
        b3 = pm.Normal('b3', mu=0, sigma=100)
        b4 = pm.Normal('b4', mu=0, sigma=100)
        b5 = pm.Normal('b5', mu=0, sigma=100)
        b6 = pm.Normal('b6', mu=0, sigma=100)

        tau_1 = pm.Gamma('tau_1', alpha=0.001, beta=0.001)
        tau_2 = pm.Gamma('tau_2', alpha=0.001, beta=0.001)
        tau_3 = pm.Gamma('tau_3', alpha=0.001, beta=0.001)
        tau_4 = pm.Gamma('tau_4', alpha=0.001, beta=0.001)
        tau_5 = pm.Gamma('tau_5', alpha=0.001, beta=0.001)
        tau_6 = pm.Gamma('tau_6', alpha=0.001, beta=0.001)

        # define random parameters (variation part)
        theta1 = pm.Normal('theta1', mu=0, tau=tau_1, shape=n_obs)
        theta2 = pm.Normal('theta2', mu=0, tau=tau_2, shape=n_obs)
        theta3 = pm.Normal('theta3', mu=0, tau=tau_3, shape=n_obs)
        theta4 = pm.Normal('theta4', mu=0, tau=tau_4, shape=n_obs)
        theta5 = pm.Normal('theta5', mu=0, tau=tau_5, shape=n_obs)
        theta6 = pm.Normal('theta6', mu=0, tau=tau_6, shape=n_obs)

        # log-normal
        sigma_a = pm.Gamma('sigma_a', alpha=0.001, beta=0.001)
        a = pm.Normal('a', mu=0.01, sigma=sigma_a, shape=n_obs)

        # random effect
        mu_b = pm.Normal('mu_b', mu=0., sigma=1000)
        sigma_b = pm.Gamma('sigma_b', alpha=0.001, beta=0.001)
        b = pm.Normal('b', mu=mu_b, sigma=sigma_b, shape=n_roadway)

        lam = beta0 + \
              (b1 + theta1[idx]) * data['log_seg_length'] + \
              (b2 + theta2[idx]) * data['lane_number_1'] + \
              (b3 + theta3[idx]) * data['lane_number_2'] + \
              (b4 + theta4[idx]) * data['rural_urban'] + \
              (b5 + theta5[idx]) * data['log_volume'] + \
              (b6 + theta6[idx]) * data['std_speed'] + a[idx] + b[idx_roadway]

        # Define likelihood
        y = pm.Poisson('y', mu=np.exp(lam), observed=data['total_crashes'])

    return model


with model_factory(Combined_weekday_peak_model_data_train):
    trace = pm.sample(15000, tune=5000, progressbar=False, chains=3, cores=1, random_seed=10)

az.summary(trace, var_names=["beta0", "b1", "b2", "b3", "b4", "b5", "b6", "tau_1", "tau_2", "tau_3", "tau_4", "tau_5", "tau_6", 'sigma_a', 'mu_b', 'sigma_b'])
pm.gelman_rubin(trace)


def model_eval(trace, Combined_weekday_peak_model_data_test, vars):
    trace_df_summary = pm.summary(trace, varnames=vars)[['mean','hdi_3%','hdi_97%']]

    lam = np.exp(trace_df_summary.loc['beta0', 'mean'] + \
          trace_df_summary.loc['b1', 'mean'] * Combined_weekday_peak_model_data_test['log_seg_length'] + \
          trace_df_summary.loc['b2', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_1'] + \
          trace_df_summary.loc['b3', 'mean'] * Combined_weekday_peak_model_data_test['lane_number_2'] + \
          trace_df_summary.loc['b4', 'mean'] * Combined_weekday_peak_model_data_test['rural_urban'] + \
          trace_df_summary.loc['b5', 'mean'] * Combined_weekday_peak_model_data_test['log_volume'] + \
          trace_df_summary.loc['b6', 'mean'] * Combined_weekday_peak_model_data_test['std_speed'] +
                 trace_df_summary.loc['mu_b', 'mean']).values

    y_true = Combined_weekday_peak_model_data_test['total_crashes'].tolist()
    Combined_weekday_peak_model_data_test['y_pred'] = lam
    mae = metrics.mean_absolute_error(y_true, lam)
    rmse = (np.sqrt(metrics.mean_squared_error(y_true, lam)))

    return mae, rmse


mae, rmse = model_eval(trace, Combined_weekday_peak_model_data_test, vars=["beta0", "b1", "b2", "b3", "b4", "b5", "b6", 'mu_b'])


############################################################
# Random forest  ##########################
############################################################
x_train = Combined_weekday_peak_model_data_train[['log_seg_length', 'lane_number_1', 'lane_number_2', 'rural_urban', 'log_volume', 'std_speed']]
y_train = Combined_weekday_peak_model_data_train['total_crashes']

x_test = Combined_weekday_peak_model_data_test[['log_seg_length', 'lane_number_1', 'lane_number_2', 'rural_urban', 'log_volume', 'std_speed']]
y_test = Combined_weekday_peak_model_data_test['total_crashes']


x_train = Combined_weekday_peak_model_data_train[['lane_number_1', 'lane_number_2', 'rural_urban', 'log_volume', 'std_speed']]
y_train = Combined_weekday_peak_model_data_train['total_crashes_offset']

x_test = Combined_weekday_peak_model_data_test[['lane_number_1', 'lane_number_2', 'rural_urban', 'log_volume', 'std_speed']]
y_test = Combined_weekday_peak_model_data_test['total_crashes_offset']


rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
rf.fit(x_train, y_train)
predictions = rf.predict(x_test)

random_grid = {'bootstrap': [True, False],
     'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10],
     'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(x_train, y_train)
predictions = rf_random.best_estimator_.predict(x_test)
# predictions_r = predictions * np.exp(Combined_weekday_peak_model_data_test['log_seg_length'])


def evaluation_metrics(test, predictions):
    rmse = (np.sqrt(metrics.mean_squared_error(test, predictions)))

    mae_sum = sum(abs(test - predictions))
    mae = mae_sum / len(test)

    return rmse, mae


rmse, mae = evaluation_metrics(y_test, predictions)


############################################################
# XGBoost  ##########################
############################################################
estimator = xgb.XGBRegressor(
    objective='reg:linear',
    nthread=4,
    seed=42
)

parameters = {'min_child_weight':[4,6,8,10],
          'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
          'gamma': [0, 1, 5, 9],
          'subsample':[i/10.0 for i in range(8,11)],
          'colsample_bytree':[i/10.0 for i in range(8,11)],
          'max_depth': [3, 4, 5, 6, 8, 10],
          'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}

random_search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=parameters,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    cv=5,
    verbose=2,
    random_state=42
)

random_search.fit(x_train, y_train)
y_pred = random_search.best_estimator_.predict(x_test)
random_search.best_estimator_

rmse, mae = evaluation_metrics(y_test, y_pred)

############################################################
# SVR  ##########################
############################################################
parameters = {'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 0.01, 0.1, 0.5, 0.9], 'C': [1, 10, 100]}

estimator = SVR(epsilon=0.01)

random_search = RandomizedSearchCV(
    estimator=estimator,
    param_distributions=parameters,
    n_iter=10,
    scoring='neg_mean_absolute_error',
    n_jobs=-1,
    cv=5,
    verbose=1
)

random_search.fit(x_train, y_train)
y_pred = random_search.best_estimator_.predict(x_test)

rmse, mae = evaluation_metrics(y_test, y_pred)

