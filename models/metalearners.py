import numpy as np
import pandas as pd
from scipy.stats import norm, beta
from sklearn.ensemble import RandomForestRegressor
from quantile_forest import RandomForestQuantileRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

from models.drlearner import *



def conformal_metalearner_experiment_jackknife_plus(df, metalearner="DR", quantile_regression=True, alpha=0.1, test_frac=0.1):
    """
    Jackknife+ version of conformal metalearner experiment that uses leave-one-out cross-validation
    instead of splitting data into train/calibration sets.
    """
    
    if len(df)==2:
        train_data1, test_data = df
    else:
        train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    # For Jackknife+, we use all training data for LOO cross-validation
    X_train_all = train_data1.filter(like='X').values
    T_train_all = train_data1[['T']].values.reshape((-1,)) 
    Y_train_all = train_data1[['Y']].values.reshape((-1,))
    ps_train_all = train_data1[['ps']].values
    
    X_test = test_data.filter(like='X').values
    T_test = test_data[['T']].values.reshape((-1,)) 
    Y_test = test_data[['Y']].values.reshape((-1,))
    ps_test = test_data[['ps']].values
    
    n_train = len(X_train_all)
    
    # Store LOO predictions and conformity scores
    loo_predictions_lower = []
    loo_predictions_upper = []
    loo_predictions_point = []
    conformity_scores = []
    
    # Leave-one-out cross-validation
    for i in range(n_train):
        # Create training indices (all except i)
        train_indices = list(range(n_train))
        train_indices.pop(i)
        
        # Get training data (all except point i)
        X_train = X_train_all[train_indices]
        T_train = T_train_all[train_indices]
        Y_train = Y_train_all[train_indices]
        ps_train = ps_train_all[train_indices]
        
        # Get left-out point for conformity score calculation
        X_left_out = X_train_all[i:i+1]
        T_left_out = T_train_all[i:i+1]
        Y_left_out = Y_train_all[i:i+1]
        ps_left_out = ps_train_all[i:i+1]
        
        # Train model on n-1 points
        model = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, 
                                    metalearner=metalearner,
                                    jackknife_plus=True)
        model.fit(X_train, T_train, Y_train, ps_train)
        
        # Get predictions for test set using this LOO model
        T_hat, T_hat_l, T_hat_u = model.predict(X_test)
        loo_predictions_lower.append(T_hat_l)
        loo_predictions_upper.append(T_hat_u)
        loo_predictions_point.append(T_hat)
        
        # Compute conformity score for left-out point
        T_hat_left_out, T_hat_l_left_out, T_hat_u_left_out = model.predict(X_left_out)
        
        # Get pseudo-outcome for left-out point
        y_hat_0_left_out = []
        y_hat_1_left_out = []
        for j in range(len(model.models_0)):
            y_hat_0_left_out.append(model.models_0[j].predict(X_left_out))
            y_hat_1_left_out.append(model.models_1[j].predict(X_left_out))
        
        y_hat_0_left_out = np.mean(np.array(y_hat_0_left_out), axis=0).reshape((-1,))
        y_hat_1_left_out = np.mean(np.array(y_hat_1_left_out), axis=0).reshape((-1,))
        
        Y_DR_left_out = model.get_pseudo_outcomes(T_left_out, ps_left_out, Y_left_out, 
                                                 y_hat_0_left_out, y_hat_1_left_out, metalearner)
        
        # Compute conformity score
        if quantile_regression:
            conformity_score = np.maximum(T_hat_l_left_out.reshape((-1,1)) - Y_DR_left_out, 
                                        Y_DR_left_out - T_hat_u_left_out.reshape((-1,1)))
        else:
            conformity_score = np.abs(T_hat_left_out.reshape((-1,1)) - Y_DR_left_out)
        
        conformity_scores.append(conformity_score.flatten())
    
    # Convert to numpy arrays
    loo_predictions_lower = np.array(loo_predictions_lower)
    loo_predictions_upper = np.array(loo_predictions_upper)
    loo_predictions_point = np.array(loo_predictions_point)
    conformity_scores = np.array(conformity_scores)
    
    # Compute Jackknife+ intervals
    n_test = len(X_test)
    T_hat_DR_l = np.zeros(n_test)
    T_hat_DR_u = np.zeros(n_test)
    T_hat_DR = np.mean(loo_predictions_point, axis=0)
    
    for j in range(n_test):
        # For each test point, compute quantiles of predictions + conformity scores
        lower_vals = loo_predictions_lower[:, j] - conformity_scores.flatten()
        upper_vals = loo_predictions_upper[:, j] + conformity_scores.flatten()
        
        # Compute quantiles
        T_hat_DR_l[j] = np.quantile(lower_vals, alpha/2)
        T_hat_DR_u[j] = np.quantile(upper_vals, 1 - alpha/2)
    
    # Compute metrics
    True_effects = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE = test_data[['CATE']].values
    
    conditional_coverage = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE = np.sqrt(np.mean((CATE-T_hat_DR)**2))
    
    # For compatibility, return conformity scores (using the last model's residuals)
    meta_conformity_score = conformity_scores[-1].reshape((-1, 1))
    oracle_conformity_score = np.abs(T_hat_DR.reshape((-1, 1)) - True_effects.reshape((-1, 1)))
    
    conformity_scores = (meta_conformity_score, oracle_conformity_score)
    
    return conditional_coverage, average_interval_width, PEHE, conformity_scores


def conformal_metalearner_experiment(df, metalearner="DR", quantile_regression=True, alpha=0.1, test_frac=0.1, use_jackknife_plus=False):
    
    # If Jackknife+ is requested, use the Jackknife+ version
    if use_jackknife_plus:
        return conformal_metalearner_experiment_jackknife_plus(df, metalearner, quantile_regression, alpha, test_frac)
    
    if len(df)==2:
        
        train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, 
                                    metalearner=metalearner) 
    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)
    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores


def dr_cqr_random_forests(df, alpha):

    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", 
                                    quantile_regression=quantile_regression, metalearner="DR") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores


def ipw_cqr_random_forests(df, alpha):
    
    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", quantile_regression=True, metalearner="IPW") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores




def x_cqr_random_forests(df, alpha):
    
    if len(df)==2:

      train_data1, test_data = df
    
    else:
    
      train_data1, test_data = train_test_split(df, test_size=test_frac, random_state=42)
    
    train_data, calib_data = train_test_split(train_data1, test_size=0.25, random_state=42)

    #X_train  = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_train  = train_data.filter(like = 'X').values
    T_train  = train_data[['T']].values.reshape((-1,)) 
    Y_train  = train_data[['Y']].values.reshape((-1,))
    ps_train = train_data[['ps']].values

    #X_calib  = calib_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_calib  = calib_data.filter(like = 'X').values
    T_calib  = calib_data[['T']].values.reshape((-1,)) 
    Y_calib  = calib_data[['Y']].values.reshape((-1,))
    ps_calib = calib_data[['ps']].values

    ITEcalib = calib_data[['Y1']].values.reshape((-1,)) - calib_data[['Y0']].values.reshape((-1,))

    #X_test   = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9', 'X10']].values
    X_test   = test_data.filter(like = 'X').values
    T_test   = test_data[['T']].values.reshape((-1,)) 
    Y_test   = test_data[['Y']].values.reshape((-1,))
    ps_test  = test_data[['ps']].values

    model    = conformalMetalearner(alpha=alpha, base_learner="GBM", quantile_regression=True, metalearner="X") 

    model.fit(X_train, T_train, Y_train, ps_train)
    model.conformalize(alpha, X_calib, T_calib, Y_calib, ps_calib, oracle=ITEcalib)

    T_hat_DR, T_hat_DR_l, T_hat_DR_u = model.predict(X_test)

    True_effects           = test_data[['Y1']].values.reshape((-1,)) - test_data[['Y0']].values.reshape((-1,))
    CATE                   = test_data[['CATE']].values

    conditional_coverage   = np.mean((True_effects >= T_hat_DR_l) & (True_effects <= T_hat_DR_u))
    average_interval_width = np.mean(np.abs(T_hat_DR_u - T_hat_DR_l))
    PEHE                   = np.sqrt(np.mean((CATE-T_hat_DR)**2))

    meta_conformity_score, oracle_conformity_score = model.residuals, model.oracle_residuals

    conformity_scores = (meta_conformity_score, oracle_conformity_score)

    return conditional_coverage, average_interval_width, PEHE, conformity_scores



def run(data, func, **kwargs): # alpha):

    results = []
  
    if type(data)==tuple:
        
        for df_train, df_test in zip(data[0], data[1]):
            
            result = func((df_train, df_test), **kwargs)# alpha)
            results.append(result)

    else:

        for df in data:
            
            result = func(df, **kwargs)# alpha)
            results.append(result)
  
    return results

