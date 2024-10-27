import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.regression.rolling import RollingOLS
from sklearn.linear_model import LinearRegression

def return_metrics(data, portfolio = None, annual_factor = 1):
    if portfolio is None:
        returns = data
    else:
        returns = data @ portfolio

    summary_statistics = pd.DataFrame(index=returns.columns)
    summary_statistics['Mean'] = returns.mean() * annual_factor
    summary_statistics['Vol'] = returns.std() * np.sqrt(annual_factor)
    summary_statistics['Sharpe'] = (returns.mean() / returns.std()) * np.sqrt(annual_factor)
    summary_statistics['Min'] = returns.min()
    summary_statistics['Max'] = returns.max()
    
    return summary_statistics



def maximumDrawdown(returns):
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max

    max_drawdown = drawdown.min()
    end_date = drawdown.idxmin()
    summary = pd.DataFrame({'Max Drawdown': max_drawdown, 'Bottom': end_date})

    for col in drawdown:
        summary.loc[col,'Peak'] = (rolling_max.loc[:end_date[col],col]).idxmax()
        recovery = (drawdown.loc[end_date[col]:,col])
        try:
            summary.loc[col,'Recover'] = pd.to_datetime(recovery[recovery >= 0].index[0])
        except:
            summary.loc[col,'Recover'] = summary.loc[col,'Recover'] = pd.NaT

        summary['Peak'] = pd.to_datetime(summary['Peak'])
        try:
            summary['Duration (to Recover)'] = (summary['Recover'] - summary['Peak'])
        except:
            summary['Duration (to Recover)'] = None
            
        summary = summary[['Max Drawdown','Peak','Bottom','Recover','Duration (to Recover)']]

    return summary    




def risk_metrics(data, portfolio = None, quantile=.05, relative=False, mdd=True):
    if portfolio is None:
        returns = data
    else:
        returns = data @ portfolio
        
    metrics = pd.DataFrame(index=returns.columns)
    metrics['Skewness'] = returns.skew()
    metrics['Kurtosis'] = returns.kurtosis()

    VaR = returns.quantile(quantile)
    CVaR = (returns[returns < returns.quantile(quantile)]).mean()

    if relative:
        VaR = (VaR - returns.mean())/returns.std()
        CVaR = (CVaR - returns.mean())/returns.std()

    metrics[f'VaR ({quantile})'] = VaR
    metrics[f'CVaR ({quantile})'] = CVaR

    if mdd:
        mdd_stats = maximumDrawdown(returns)
        metrics = metrics.join(mdd_stats)

        if relative:
            metrics['Max Drawdown'] = (metrics['Max Drawdown'] - returns.mean())/returns.std()

    return metrics

def get_ols_metrics(regressors, targets, annualization=1, ignorenan=True, intercept=True):
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    Xset = df_aligned[regressors.columns]

    reg = pd.DataFrame(index=targets.columns)

    for col in Y.columns:
        y = Y[col]

        if ignorenan:
            alldata = Xset.join(y, lsuffix='X')
            mask = alldata.notnull().all(axis=1)
            y = y[mask]
            X = Xset[mask]
        else:
            X = Xset

        if intercept:
            X = sm.add_constant(X)  # Add intercept (constant term)

        model = sm.OLS(y, X).fit()

        # Store the regression results
        if intercept:
            reg.loc[col, 'alpha'] = model.params['const'] * annualization  # Intercept
        else:
            reg.loc[col, 'alpha'] = 0  # No intercept (alpha is 0)
        
        # Store coefficients
        for column in regressors.columns:
            reg.loc[col, "Beta: " + column] = model.params[column]
        
        # Store R-squared
        reg.loc[col, 'r-squared'] = model.rsquared

        # Calculate residuals
        residuals = model.resid

        # Calculate volatility of residuals (standard deviation)
        residual_volatility = residuals.std() * np.sqrt(annualization)
        reg.loc[col, 'Volatility of Residuals'] = residual_volatility

        # Treynor Ratio is only defined for univariate regression
        if Xset.shape[1] == 1:
            if np.abs(model.params[regressors.columns[0]]) < 1e-12:
                reg.loc[col, 'Treynor Ratio'] = None
            else:
                reg.loc[col, 'Treynor Ratio'] = (y.mean() / model.params[regressors.columns[0]]) * annualization

        # Information Ratio
        if intercept:
            if np.abs(model.params['const']) < 1e-12:
                reg.loc[col, 'Info Ratio'] = None
            else:
                reg.loc[col, 'Info Ratio'] = (model.params['const'] / residuals.std()) * np.sqrt(annualization)
        else:
            reg.loc[col, 'Info Ratio'] = None  # No intercept means no information ratio

    return reg




def display_correlation(df,annot=True,list_maxmin=True):
    
    corrmat = df.corr()
    #ignore self-correlation
    corrmat[corrmat==1] = None
    sns.heatmap(corrmat,annot=annot,fmt='.0%')

    if list_maxmin:
        corr_rank = corrmat.unstack().sort_values().dropna()
        pair_max = corr_rank.index[-1]
        pair_min = corr_rank.index[0]

        print(f'MIN Correlation pair is {pair_min}')
        print(f'MAX Correlation pair is {pair_max}')
        
    return


def get_rolling_ols_metrics(regressors, targets, window=60, annualization=1, ignorenan=True, intercept=True):
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    Xset = df_aligned[regressors.columns]

    replication = targets.copy()

    y = Y.squeeze()  # Assuming it's a series (single target column)

    if ignorenan:
        alldata = Xset.join(y, lsuffix='X')
        mask = alldata.notnull().all(axis=1)
        y = y[mask]
        X = Xset[mask]
    else:
        X = Xset

    ### 1. Rolling Betas and Replication with Intercept ###
    if intercept:
        X_with_intercept = sm.add_constant(X)  # Add intercept

        # Rolling OLS with intercept
        rolling_model_with_intercept = RollingOLS(y, X_with_intercept, window=window)
        rolling_betas_with_intercept = rolling_model_with_intercept.fit().params.copy()

        # In-sample replication (with intercept)
        rep_IS_with_intercept = (rolling_betas_with_intercept * X_with_intercept).sum(axis=1, skipna=False)

        # Out-of-sample replication (with intercept)
        rep_OOS_with_intercept = (rolling_betas_with_intercept.shift() * X_with_intercept).sum(axis=1, skipna=False)

        # Add replication results to the DataFrame
        replication['Rolling-IS-Int'] = rep_IS_with_intercept
        replication['Rolling-OOS-Int'] = rep_OOS_with_intercept

    ### 2. Rolling Betas and Replication without Intercept ###
    rolling_model_without_intercept = RollingOLS(y, X, window=window)
    rolling_betas_without_intercept = rolling_model_without_intercept.fit().params.copy()

    # In-sample replication (without intercept)
    rep_IS_without_intercept = (rolling_betas_without_intercept * X).sum(axis=1, skipna=False)

    # Out-of-sample replication (without intercept)
    rep_OOS_without_intercept = (rolling_betas_without_intercept.shift() * X).sum(axis=1, skipna=False)

    # Add replication results to the DataFrame
    replication['Rolling-IS-NoInt'] = rep_IS_without_intercept
    replication['Rolling-OOS-NoInt'] = rep_OOS_without_intercept

    ### 3. Static OLS with and without intercept (for comparison) ###
    if intercept:
        # Static OLS with intercept
        static_model_with_intercept = sm.OLS(y, X_with_intercept).fit()
        replication['Static-IS-Int'] = static_model_with_intercept.fittedvalues

    # Static OLS without intercept
    static_model_without_intercept = sm.OLS(y, X).fit()
    replication['Static-IS-NoInt'] = static_model_without_intercept.fittedvalues

    ### 4. Performance Metrics ###
    def calculate_metrics(true_values, predicted_values):
        # Calculate R²
        ss_residual = ((true_values - predicted_values) ** 2).sum()
        ss_total = ((true_values - true_values.mean()) ** 2).sum()
        r_squared = 1 - ss_residual / ss_total

        # Correlation
        correlation = true_values.corr(predicted_values)

        # Mean Error
        mean_error = (true_values - predicted_values).mean()

        return r_squared, correlation, mean_error

    # Store metrics for each replication method
    performance_metrics = {}

    if intercept:
        # Metrics for IS with intercept
        performance_metrics['R²_IS_Int'], performance_metrics['Corr_IS_Int'], performance_metrics['MeanError_IS_Int'] = \
            calculate_metrics(y, replication['Rolling-IS-Int'])

        # Metrics for OOS with intercept
        performance_metrics['R²_OOS_Int'], performance_metrics['Corr_OOS_Int'], performance_metrics['MeanError_OOS_Int'] = \
            calculate_metrics(y, replication['Rolling-OOS-Int'])

    # Metrics for IS without intercept
    performance_metrics['R²_IS_NoInt'], performance_metrics['Corr_IS_NoInt'], performance_metrics['MeanError_IS_NoInt'] = \
        calculate_metrics(y, replication['Rolling-IS-NoInt'])

    # Metrics for OOS without intercept
    performance_metrics['R²_OOS_NoInt'], performance_metrics['Corr_OOS_NoInt'], performance_metrics['MeanError_OOS_NoInt'] = \
        calculate_metrics(y, replication['Rolling-OOS-NoInt'])

    ### 5. Static OLS Metrics ###
    if intercept:
        performance_metrics['R²_Static_IS_Int'], performance_metrics['Corr_Static_IS_Int'], performance_metrics['MeanError_Static_IS_Int'] = \
            calculate_metrics(y, replication['Static-IS-Int'])

    performance_metrics['R²_Static_IS_NoInt'], performance_metrics['Corr_Static_IS_NoInt'], performance_metrics['MeanError_Static_IS_NoInt'] = \
        calculate_metrics(y, replication['Static-IS-NoInt'])

    return (rolling_betas_with_intercept if intercept else None, rolling_betas_without_intercept, replication, performance_metrics)


def get_nnls_estimates(regressors, targets, annualization=1, ignorenan=True):
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    # Align targets and regressors
    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    Xset = df_aligned[regressors.columns]

    reg = pd.DataFrame(index=targets.columns)

    for col in Y.columns:
        y = Y[col]

        if ignorenan:
            # Only use non-NaN data
            alldata = Xset.join(y, lsuffix='X')
            mask = alldata.notnull().all(axis=1)
            y = y[mask]
            X = Xset[mask]
        else:
            X = Xset

        # Fit non-negative least squares regression
        model = LinearRegression(positive=True).fit(X, y)
        reg.loc[col, 'alpha'] = model.intercept_ * annualization
        reg.loc[col, regressors.columns] = model.coef_
        reg.loc[col, 'r-squared'] = model.score(X, y)

    return reg


def get_glm_estimates_with_constraints(regressors, targets, bounds=None, annualization=1, ignorenan=True):
    if not isinstance(regressors, pd.DataFrame):
        regressors = regressors.to_frame()
    if not isinstance(targets, pd.DataFrame):
        targets = targets.to_frame()

    # Align targets and regressors
    df_aligned = targets.join(regressors, how='inner', lsuffix='y ')
    Y = df_aligned[targets.columns]
    Xset = df_aligned[regressors.columns]

    reg = pd.DataFrame(index=targets.columns)

    for col in Y.columns:
        y = Y[col]

        if ignorenan:
            # Only use non-NaN data
            alldata = Xset.join(y, lsuffix='X')
            mask = alldata.notnull().all(axis=1)
            y = y[mask]
            X = Xset[mask]
        else:
            X = Xset

        # Add intercept
        X_with_intercept = sm.add_constant(X)

        # Fit GLM
        model = sm.GLM(y, X_with_intercept).fit()

        # Apply bounds to betas (if bounds are given)
        if bounds is not None:
            for idx, bound in enumerate(bounds):
                lower_bound, upper_bound = bound
                model.params[idx] = max(lower_bound, min(model.params[idx], upper_bound))

        reg.loc[col, 'alpha'] = model.params['const'] * annualization
        reg.loc[col, regressors.columns] = model.params[regressors.columns]
        # reg.loc[col, 'r-squared'] = model.rsquared

    return reg

def calculate_historic_expanding_var(data, start_date='2001-01-02', quantile=0.05, min_window_size=60):
    var_df = pd.DataFrame(index=data.index)
    under_var_frequency = {}
    for asset in data.columns:
        shifted_returns = data[asset].shift()
        var_df[f'VaR_{asset}'] = shifted_returns.expanding(min_periods=min_window_size).quantile(quantile)
        asset_var_df = var_df.loc[start_date:][f'VaR_{asset}']
        asset_var_df.plot(title=f'Historic Expanding VaR for {asset} (Quantile={quantile})', ylabel='VaR', xlabel='Date')
        plt.show()
        actual_returns = data[asset].loc[start_date:]
        var_df[f'Below_VaR_{asset}'] = actual_returns < var_df[f'VaR_{asset}'].loc[start_date:]
        under_var_frequency[asset] = var_df[f'Below_VaR_{asset}'].mean()

    return var_df, under_var_frequency

def calculate_historic_rolling_var(data, start_date='2001-01-02', quantile=0.05, window_size=60):
    var_df = pd.DataFrame(index=data.index)
    under_var_frequency = {}
    for asset in data.columns:
        shifted_returns = data[asset].shift()
        var_df[f'VaR_{asset}'] = shifted_returns.rolling(window=window_size).quantile(quantile)
        asset_var_df = var_df.loc[start_date:][f'VaR_{asset}']
        asset_var_df.plot(title=f'Historic Rolling VaR for {asset} (Quantile={quantile})', ylabel='VaR', xlabel='Date')
        plt.show()
        actual_returns = data[asset].loc[start_date:]
        var_df[f'Below_VaR_{asset}'] = actual_returns < var_df[f'VaR_{asset}'].loc[start_date:]
        under_var_frequency[asset] = var_df[f'Below_VaR_{asset}'].mean()
    return var_df, under_var_frequency

def calculate_expanding_vol(return_series: pd.Series, window_size=252) -> pd.Series:
    return np.sqrt((return_series ** 2).expanding(window_size).mean())

def calculate_rolling_vol(return_series: pd.Series, window_size=252) -> pd.Series:
    return np.sqrt((return_series ** 2).rolling(window_size).mean())

def calculate_ewma_volatility(data: pd.Series, theta : float = 0.94, initial_vol : float = .2 / np.sqrt(252), window_size=252) -> pd.Series:
    ewma = np.zeros(len(data))
    ewma[0] = initial_vol
    for t in range(1, len(data)):
        ewma[t] = np.sqrt(theta * ewma[t-1]**2 + (1 - theta) * data.iloc[t-1]**2)
    return ewma


def calculate_volatility_var(
    data,
    ewma_theta=0.94,
    ewma_initial_vol=.2/np.sqrt(252),
    window_size=252, 
    z_score=-1.65, 
    start_date='2001-01-02'
):
    VaR_df = pd.DataFrame(index=data.index)

    for asset in data.columns:
        returns = data[asset]
        expanding_vol = calculate_expanding_vol(returns, window_size)
        
        rolling_vol = calculate_rolling_vol(returns, window_size)
        
        ewma_vol = calculate_ewma_volatility(returns, ewma_theta, ewma_initial_vol, window_size)

        VaR_df[f'Expanding Volatility VaR_{asset}'] = z_score * expanding_vol
        VaR_df[f'Rolling Volatility VaR_{asset}'] = z_score * rolling_vol
        VaR_df[f'EWMA Volatility VaR_{asset}'] = z_score * ewma_vol

        VaR_df.loc[start_date:, [f'Expanding Volatility VaR_{asset}', 
                                 f'Rolling Volatility VaR_{asset}', 
                                 f'EWMA Volatility VaR_{asset}']].plot(title=f'Volatility-Based VaR Estimates for {asset}', ylabel='VaR', xlabel='Date')
        plt.show()

    return VaR_df

def calculate_var_violations(var_series: pd.Series, excess_returns: pd.Series, expected_quantile: float = 0.05):
    violations = (excess_returns < var_series).sum()
    
    total_periods = len(excess_returns)
    hit_ratio = violations / total_periods
    
    hit_error = (hit_ratio/expected_quantile)-1
    
    return violations, hit_ratio, hit_error

def historical_expanding_cvar(return_series: pd.Series, percentile: float = .05) -> pd.Series:
    return return_series.expanding(252).apply(lambda x: x[x < x.quantile(percentile)].mean())

def historical_rolling_cvar(return_series: pd.Series, percentile: float = .05) -> pd.Series:
    return return_series.rolling(252).apply(lambda x: x[x < x.quantile(percentile)].mean())
