"""This ARIMA Box-Jenkins method is based on
Box, G.E.P. and Jenkins, G.M. (1970) Time Series Analysis: Forecasting and Control, San Francisco: Holden-Day.
The reason for this method is to enable most accurate ARIMA results possible according to the established 3 stage method.
Box-Jenkins Method has 3 stages: Identification, Estimation, Diagnosis Checking.
This code was made by collaboration of research based prompting using ChatGPT under the guidance from but not inclusive of:

                                                            1\. Box, G.E.P. and Jenkins, G.M. (1976) *Time Series Analysis: Forecasting and Control*. 2nd edn. San Francisco: Holden-Day.

                                                            2\. Young, W\.L. (1977) ‘The Box-Jenkins approach to time series analysis and forecasting: principles and applications’, *RAIRO. Recherche opérationnelle*, 11(2), pp. 129–143.

                                                            3\. NCSS Statistical Software (n.d.) ‘The Box-Jenkins Method’, Chapter 470, *NCSS User’s Guide IV*.

                                                            4\. GeeksforGeeks (2025) ‘Box-Jenkins Methodology for ARIMA Models’.

                                                            5\. Brownlee, J. (n.d.) ‘A Gentle Introduction to the Box-Jenkins Method for Time Series Forecasting’. *Machine Learning Mastery*."""


import warnings
import numpy as np
import pandas as pd
import yfinance as yf

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from sklearn.metrics import mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# =====================================================
# 1. CONFIGURATION
# =====================================================

TICKERS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "GOOGL",
    "META", "AVGO", "AMD", "ORCL", "CSCO"
]

START_DATE = "2025-11-12"
END_DATE = "2026-11-12"

PREDICT_TIME = 5
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15

SIGNIFICANCE_LEVEL = 0.05
MAX_LAGS = 15

# Controlled candidate model range
P_VALUES = [0, 1, 2]
Q_VALUES = [0, 1, 2]

# DFO full-stock hyperparameter grid: 14 populations x 3 disturbance rates.
DFO_POPULATION_SIZES = [
    10, 20, 30, 40, 50
]
DFO_DISTURBANCE_RATES = [0.1, 0.01, 0.001]
DFO_SEARCH_RADIUS = 2.5
DFO_RUN_SEEDS = [1, 2, 3, 4, 5]
DFO_MAX_ITERATIONS = 1500
DFO_PATIENCE = 10
DFO_MIN_IMPROVEMENT = 1e-6
DFO_ORIGIN_STRIDE = 10


# =====================================================
# 2. DOWNLOAD AND PREPARE DATA
# =====================================================

prices = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)["Close"]

prices = prices.dropna(how="all")

# Daily logarithmic returns
returns = np.log(prices / prices.shift(1)).dropna()

# Five-day forward realised variance
realised_variance = (
    returns.pow(2)
    .rolling(window=5)
    .sum()
    .shift(-5)
)

# Annualised five-day realised volatility
volatility = np.sqrt(
    realised_variance * (252 / 5)
).dropna()

series = {
    ticker: volatility[ticker].dropna()
    for ticker in TICKERS
}

print(
    "Planned DFO grid:",
    f"{len(DFO_POPULATION_SIZES)} populations x "
    f"{len(DFO_DISTURBANCE_RATES)} disturbance rates = "
    f"{len(DFO_POPULATION_SIZES) * len(DFO_DISTURBANCE_RATES)} configurations per stock;",
    f"{len(DFO_RUN_SEEDS)} seeds per configuration;",
    f"{len(TICKERS) * len(DFO_POPULATION_SIZES) * len(DFO_DISTURBANCE_RATES) * len(DFO_RUN_SEEDS)} "
    "planned DFO runs across all stocks."
)


# =====================================================
# 3. HELPER FUNCTIONS
# =====================================================

def run_stationarity_tests(data):

    adf_result = adfuller(
        data,
        regression="c",
        autolag="AIC"
    )

    kpss_result = kpss(
        data,
        regression="c",
        nlags="auto"
    )

    adf_p = adf_result[1]
    kpss_p = kpss_result[1]

    if adf_p < SIGNIFICANCE_LEVEL and kpss_p >= SIGNIFICANCE_LEVEL:
        conclusion = "Stationary"

    elif adf_p >= SIGNIFICANCE_LEVEL and kpss_p < SIGNIFICANCE_LEVEL:
        conclusion = "Non-stationary"

    elif adf_p < SIGNIFICANCE_LEVEL and kpss_p < SIGNIFICANCE_LEVEL:
        conclusion = "Conflicting"

    else:
        conclusion = "Inconclusive"

    return {
        "ADF_statistic": adf_result[0],
        "ADF_pvalue": adf_p,
        "KPSS_statistic": kpss_result[0],
        "KPSS_pvalue": kpss_p,
        "Conclusion": conclusion
    }


def calculate_acf_pacf(data):

    acf_values = acf(
        data,
        nlags=MAX_LAGS,
        fft=False
    )

    pacf_values = pacf(
        data,
        nlags=MAX_LAGS,
        method="ywm"
    )

    confidence_limit = 1.96 / np.sqrt(len(data))

    results = pd.DataFrame({
        "Lag": range(MAX_LAGS + 1),
        "ACF": acf_values,
        "PACF": pacf_values
    })

    results["ACF_significant"] = (
        np.abs(results["ACF"]) > confidence_limit
    )

    results["PACF_significant"] = (
        np.abs(results["PACF"]) > confidence_limit
    )

    return results, confidence_limit


def calculate_aicc(fitted_model):

    n = fitted_model.nobs
    k = len(fitted_model.params)

    if n - k - 1 <= 0:
        return np.inf

    return (
        fitted_model.aic
        + (2 * k * (k + 1)) / (n - k - 1)
    )


def calculate_metrics(actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(actual, predicted)
    )

    mae = mean_absolute_error(
        actual, predicted
    )

    return rmse, mae


def build_rolling_origin_models(
    data, target, order, gap, block_size, origin_stride=None
):
    """Prebuild the ARIMA filter models for rolling forecast origins."""
    origin_models = []
    origin_step = block_size if origin_stride is None else origin_stride

    for block_start in range(0, len(target), origin_step):
        target_block = target.iloc[block_start:block_start + block_size]
        first_target_position = data.index.get_loc(target_block.index[0])
        prefix_end = first_target_position - gap
        prefix = data.iloc[:prefix_end]

        if len(prefix) < 10:
            raise ValueError("Not enough observations before rolling forecast origin.")

        origin_models.append((
            ARIMA(prefix, order=order),
            target_block.index
        ))

    return origin_models


def rolling_origin_forecast(
    data, target, order, params, gap, block_size,
    origin_stride=None, origin_models=None
):
    """Forecast target blocks from successive chronological origins.

    At each origin, filter the observations before the purged gap, forecast
    across the gap, then retain forecasts for the next block. Model
    coefficients remain fixed across origins. Prebuilt origin models can be
    supplied to avoid reconstructing ARIMA model objects for every DFO fly.
    """
    if origin_models is None:
        origin_models = build_rolling_origin_models(
            data, target, order, gap, block_size, origin_stride
        )

    predictions = []
    prediction_dates = []
    for origin_model, block_index in origin_models:
        block_length = len(block_index)
        origin_fit = origin_model.filter(params, transformed=True)
        forecast_all = np.asarray(
            origin_fit.forecast(steps=gap + block_length), dtype=float
        )
        predictions.extend(forecast_all[gap:gap + block_length].tolist())
        prediction_dates.extend(block_index.tolist())

    return pd.Series(predictions, index=prediction_dates, name="Forecast")


def run_dfo_configuration(
    ticker, ticker_series, validation, test, selected_order, dfo_model,
    validation_origin_models, test_origin_models, population_size,
    disturbance_rate, search_radius, run_seeds, max_iterations=1500,
    patience=10, min_improvement=1e-6, gap=5, block_size=5
):
    """Run independent DFO seeds for one population/disturbance setting."""
    N = population_size
    D = len(dfo_model.param_names)
    delta = disturbance_rate
    lowerB = [-search_radius] * D
    upperB = [search_radius] * D
    objective_calls = [0]

    def f(x):
        objective_calls[0] += 1
        try:
            candidate_parameters = dfo_model.transform_params(
                np.asarray(x, dtype=float)
            )
            forecast = rolling_origin_forecast(
                data=ticker_series,
                target=validation,
                order=selected_order,
                params=candidate_parameters,
                gap=gap,
                block_size=block_size,
                origin_models=validation_origin_models
            )
            actual = validation.loc[forecast.index]
            rmse, _ = calculate_metrics(actual, forecast)
            return float(rmse) if np.isfinite(rmse) else np.inf
        except Exception:
            return np.inf

    run_records = []
    for run_number, run_seed in enumerate(run_seeds, start=1):
        np.random.seed(run_seed)
        objective_calls[0] = 0
        X = np.empty([N, D])
        fitness = [None] * N

        # Original DFO population initialization, bounded per dimension.
        for i in range(N):
            for d in range(D):
                X[i, d] = np.random.uniform(lowerB[d], upperB[d])

        # Original DFO loop: evaluation, best fly, elitism, ring neighbours,
        # coordinate updates, disturbance and random boundary resets.
        best_fitness_history = []
        best_fitness_so_far = np.inf
        iterations_without_improvement = 0
        iterations_completed = 0
        convergence_iteration = None

        for itr in range(max_iterations):
            for i in range(N):
                fitness[i] = f(X[i,])
            s = np.argmin(fitness)
            current_best = float(fitness[s])
            best_fitness_history.append(current_best)
            iterations_completed = itr + 1

            if (not np.isfinite(best_fitness_so_far)
                    or current_best < best_fitness_so_far - min_improvement):
                best_fitness_so_far = current_best
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1

            if iterations_without_improvement >= patience:
                convergence_iteration = iterations_completed
                break

            for i in range(N):
                if i == s:
                    continue
                left = (i - 1) % N
                right = (i + 1) % N
                bNeighbour = right if fitness[right] < fitness[left] else left

                for d in range(D):
                    if np.random.rand() < delta:
                        X[i, d] = np.random.uniform(lowerB[d], upperB[d])
                        continue
                    u = np.random.rand()
                    X[i, d] = X[bNeighbour, d] + u * (X[s, d] - X[i, d])
                    if X[i, d] < lowerB[d] or X[i, d] > upperB[d]:
                        X[i, d] = np.random.uniform(lowerB[d], upperB[d])

        # Evaluate the final population and apply its best candidate directly.
        for i in range(N):
            fitness[i] = f(X[i,])
        s = np.argmin(fitness)
        unconstrained_parameters = np.asarray(X[s,], dtype=float).copy()
        dfo_parameters = dfo_model.transform_params(unconstrained_parameters)
        dfo_forecast = rolling_origin_forecast(
            data=ticker_series,
            target=test,
            order=selected_order,
            params=dfo_parameters,
            gap=gap,
            block_size=block_size,
            origin_models=test_origin_models
        )
        test_rmse, test_mae = calculate_metrics(test, dfo_forecast)
        convergence_status = (
            f"patience stopping at iteration {convergence_iteration}"
            if convergence_iteration is not None
            else f"maximum iteration limit ({max_iterations}) reached"
        )

        record = {
            "Ticker": ticker,
            "Population_Size": N,
            "Disturbance_Rate": delta,
            "Search_Radius": search_radius,
            "Parameter_Dimension": D,
            "Parameter_Names": str(dfo_model.param_names),
            "Run": run_number,
            "Seed": run_seed,
            "Validation_RMSE": float(fitness[s]),
            "Test_RMSE": float(test_rmse),
            "Test_MAE": float(test_mae),
            "Objective_Evaluations": objective_calls[0],
            "Iterations_Completed": iterations_completed,
            "Convergence_Iteration": convergence_iteration,
            "Convergence_Status": convergence_status,
            "Parameters": str(dfo_parameters),
            "Forecast": dfo_forecast.copy(),
            "Best_Fitness_By_Iteration": str(best_fitness_history)
        }
        run_records.append(record)
        print(
            f"{ticker} | N={N} | delta={delta:g} | radius={search_radius:g} | "
            f"run {run_number}/{len(run_seeds)} seed={run_seed} | "
            f"iterations={iterations_completed} | objective evaluations={objective_calls[0]} | "
            f"{convergence_status} | validation RMSE={fitness[s]:.6f} | "
            f"test RMSE={test_rmse:.6f} | test MAE={test_mae:.6f}"
        )

    return run_records


# =====================================================
# 4. BOX-JENKINS ANALYSIS
# =====================================================

model_results = []
configuration_results = []
forecast_results = []
monte_carlo_results = []
diagnostic_results = []
identification_results = []

for ticker, s in series.items():
    # Preserve the ticker's time series under a separate name. The original
    # DFO loop uses `s` for the best-fly index, so it must not replace the data.
    ticker_series = s

    print("\n" + "=" * 65)
    print("STOCK:", ticker)
    print("=" * 65)

    n = len(s)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    # Five-observation purge at partition boundaries
    train = s.iloc[:train_end - PREDICT_TIME]

    validation = s.iloc[
        train_end:val_end - PREDICT_TIME
    ]

    test = s.iloc[val_end:]

    print("Training observations:", len(train))
    print("Validation observations:", len(validation))
    print("Testing observations:", len(test))

    if len(train) <= MAX_LAGS + 5:
        print("Insufficient training observations.")
        continue

    # -------------------------------------------------
    # STAGE 1: IDENTIFICATION
    # -------------------------------------------------

    print("\nSTAGE 1: IDENTIFICATION")

    # Original series
    level_stationarity = run_stationarity_tests(train)

    # First difference
    differenced = train.diff().dropna()

    diff_stationarity = run_stationarity_tests(differenced)

    print("\nOriginal series stationarity:")
    print(level_stationarity)

    print("\nFirst-differenced series stationarity:")
    print(diff_stationarity)

    # ACF/PACF of original series
    level_acf_pacf, level_limit = calculate_acf_pacf(train)

    # ACF/PACF of first-differenced series
    diff_acf_pacf, diff_limit = calculate_acf_pacf(differenced)

    print("\nOriginal series ACF/PACF:")
    print(level_acf_pacf.round(4).to_string(index=False))

    print("\nDifferenced series ACF/PACF:")
    print(diff_acf_pacf.round(4).to_string(index=False))

    # Determine candidate differencing orders
    if level_stationarity["Conclusion"] == "Stationary":

        candidate_d = [0]

    elif level_stationarity["Conclusion"] == "Non-stationary":

        candidate_d = [1]

    else:

        # Conflicting/inconclusive evidence:
        # investigate both specifications
        candidate_d = [0, 1]

    print("\nCandidate differencing orders:", candidate_d)

    identification_results.append({
        "Ticker": ticker,
        "ADF_pvalue": level_stationarity["ADF_pvalue"],
        "KPSS_pvalue": level_stationarity["KPSS_pvalue"],
        "Stationarity": level_stationarity["Conclusion"],
        "Differenced_ADF_pvalue": diff_stationarity["ADF_pvalue"],
        "Differenced_KPSS_pvalue": diff_stationarity["KPSS_pvalue"],
        "Candidate_d": str(candidate_d)
    })

    # -------------------------------------------------
    # STAGE 2: PARAMETER ESTIMATION
    # -------------------------------------------------

    print("\nSTAGE 2: PARAMETER ESTIMATION")

    candidate_models = []

    for d in candidate_d:

        for p in P_VALUES:

            for q in Q_VALUES:

                order = (p, d, q)

                try:

                    model = ARIMA(
                        train,
                        order=order
                    )

                    fitted = model.fit()

                    aicc = calculate_aicc(fitted)

                    # Rolling-origin validation forecasts in five-target
                    # blocks, with the same five-observation purge per origin.
                    forecast = rolling_origin_forecast(
                        data=ticker_series,
                        target=validation,
                        order=order,
                        params=np.asarray(fitted.params, dtype=float),
                        gap=PREDICT_TIME,
                        block_size=PREDICT_TIME
                    )

                    rmse, mae = calculate_metrics(
                        validation,
                        forecast
                    )

                    candidate_models.append({
                        "order": order,
                        "fitted": fitted,
                        "AICc": aicc,
                        "Validation_RMSE": rmse,
                        "Validation_MAE": mae
                    })

                    print(
                        f"ARIMA{order} | "
                        f"AICc: {aicc:.4f} | "
                        f"Validation RMSE: {rmse:.6f}"
                    )

                except Exception as error:

                    print(
                        f"ARIMA{order} failed: {error}"
                    )

    if not candidate_models:
        print("No candidate models fitted successfully.")
        continue

    # Select by validation RMSE.
    # AICc is retained as a secondary model-selection criterion.
    candidate_models.sort(
        key=lambda x: (
            x["Validation_RMSE"],
            x["AICc"]
        )
    )

    selected = candidate_models[0]

    selected_order = selected["order"]
    selected_model = selected["fitted"]

    print("\nSelected model:", selected_order)
    print("Selected model AICc:", selected["AICc"])
    print("Validation RMSE:", selected["Validation_RMSE"])
    print("Validation MAE:", selected["Validation_MAE"])

    model_results.append({
        "Ticker": ticker,
        "Selected_ARIMA": str(selected_order),
        "AICc": selected["AICc"],
        "Validation_RMSE": selected["Validation_RMSE"],
        "Validation_MAE": selected["Validation_MAE"],
        "Number_of_candidates": len(candidate_models)
    })

    # -------------------------------------------------
    # STAGE 3: DIAGNOSTIC CHECKING
    # -------------------------------------------------

    print("\nSTAGE 3: DIAGNOSTIC CHECKING")

    residuals = pd.Series(
        selected_model.resid
    ).dropna()

    p, d, q = selected_order

    # Ljung-Box test
    ljungbox = acorr_ljungbox(
        residuals,
        lags=[5, 10, 15],
        model_df=p + q,
        return_df=True
    )

    print("\nLjung-Box results:")
    print(ljungbox)

    # ARCH-LM test
    arch_result = het_arch(
        residuals,
        nlags=5,
        ddof=p + q
    )

    arch_stat = arch_result[0]
    arch_pvalue = arch_result[1]

    print("\nARCH-LM statistic:", arch_stat)
    print("ARCH-LM p-value:", arch_pvalue)

    # Residual ACF/PACF
    residual_acf_pacf, residual_limit = calculate_acf_pacf(
        residuals
    )

    print("\nResidual ACF/PACF:")
    print(residual_acf_pacf.round(4).to_string(index=False))

    # Coefficient estimates and significance
    print("\nEstimated model parameters:")
    print(selected_model.summary())

    diagnostic_results.append({
        "Ticker": ticker,
        "ARIMA": str(selected_order),
        "LjungBox_p_lag5": ljungbox["lb_pvalue"].iloc[0],
        "LjungBox_p_lag10": ljungbox["lb_pvalue"].iloc[1],
        "LjungBox_p_lag15": ljungbox["lb_pvalue"].iloc[2],
        "ARCH_statistic": arch_stat,
        "ARCH_pvalue": arch_pvalue
    })

    # -------------------------------------------------
    # STAGE 4: MODEL REFINEMENT
    # -------------------------------------------------

    print("\nSTAGE 4: MODEL REFINEMENT")

    lb_pvalue = ljungbox["lb_pvalue"].iloc[0]

    if lb_pvalue < SIGNIFICANCE_LEVEL:

        print(
            "Warning: residual autocorrelation remains "
            "at lag 5. Model may be inadequate."
        )

    else:

        print(
            "No significant residual autocorrelation "
            "detected at lag 5."
        )

    if arch_pvalue < SIGNIFICANCE_LEVEL:

        print(
            "ARCH effects remain in residuals."
        )

    else:

        print(
            "No significant ARCH effects detected."
        )

    print(
        "Candidate models have been compared using "
        "validation performance and AICc."
    )

    # -------------------------------------------------
    # STAGE 5: FINAL TEST FORECAST
    # -------------------------------------------------

    print("\nSTAGE 5: FINAL FORECASTING")

    # Refit using all available observations before
    # the purged test boundary.
    final_train = s.iloc[:val_end - PREDICT_TIME]

    final_model = ARIMA(
        final_train,
        order=selected_order
    ).fit()

    # Prebuild rolling-origin states once; reuse them for the MLE and all DFO runs.
    test_origin_models = build_rolling_origin_models(
        data=ticker_series,
        target=test,
        order=selected_order,
        gap=PREDICT_TIME,
        block_size=PREDICT_TIME
    )

    # Evaluate MLE with the same rolling-origin protocol used for DFO:
    # fixed fitted coefficients, successive origins, and purged five-target gaps.
    final_forecast = rolling_origin_forecast(
        data=ticker_series,
        target=test,
        order=selected_order,
        params=np.asarray(final_model.params, dtype=float),
        gap=PREDICT_TIME,
        block_size=PREDICT_TIME,
        origin_models=test_origin_models
    )

    test_rmse, test_mae = calculate_metrics(
        test,
        final_forecast
    )

    print("\nFinal test RMSE:", test_rmse)
    print("Final test MAE:", test_mae)

    comparison = pd.DataFrame({
        "Ticker": ticker,
        "Actual": test,
        "MLE_Forecast": final_forecast
    })

    forecast_results.append(comparison)

    model_results[-1]["Test_RMSE"] = test_rmse
    model_results[-1]["Test_MAE"] = test_mae

    # -------------------------------------------------
    # STAGE 6: DFO HYPERPARAMETER EXPERIMENTS
    # -------------------------------------------------
    print("\nSTAGE 6: DFO HYPERPARAMETER EXPERIMENTS")

    dfo_model = ARIMA(final_train, order=selected_order)
    parameter_names = list(dfo_model.param_names)

    validation_origin_models = build_rolling_origin_models(
        data=ticker_series,
        target=validation,
        order=selected_order,
        gap=PREDICT_TIME,
        block_size=PREDICT_TIME,
        origin_stride=DFO_ORIGIN_STRIDE
    )
    print("Ticker:", ticker)
    print("Selected ARIMA order:", selected_order)
    print("Parameter names / D:", parameter_names, "/", len(parameter_names))
    print("Search bounds in unconstrained coordinates:",
          (-DFO_SEARCH_RADIUS, DFO_SEARCH_RADIUS))
    print("Validation origins per objective evaluation:",
          len(validation_origin_models))

    ticker_configuration_results = []
    for population_size in DFO_POPULATION_SIZES:
        for disturbance_rate in DFO_DISTURBANCE_RATES:
            print(
                f"\nStarting configuration: population={population_size}, "
                f"disturbance={disturbance_rate:g}, radius={DFO_SEARCH_RADIUS:g}"
            )
            dfo_run_records = run_dfo_configuration(
                ticker=ticker,
                ticker_series=ticker_series,
                validation=validation,
                test=test,
                selected_order=selected_order,
                dfo_model=dfo_model,
                validation_origin_models=validation_origin_models,
                test_origin_models=test_origin_models,
                population_size=population_size,
                disturbance_rate=disturbance_rate,
                search_radius=DFO_SEARCH_RADIUS,
                run_seeds=DFO_RUN_SEEDS,
                max_iterations=DFO_MAX_ITERATIONS,
                patience=DFO_PATIENCE,
                min_improvement=DFO_MIN_IMPROVEMENT,
                gap=PREDICT_TIME,
                block_size=PREDICT_TIME
            )
            monte_carlo_results.extend({
                key: value for key, value in record.items()
                if key not in {"Forecast", "Best_Fitness_By_Iteration"}
            } for record in dfo_run_records)

            # Select only using validation RMSE; never use test scores to
            # select the configuration or the displayed run.
            selected_run = min(dfo_run_records, key=lambda r: r["Validation_RMSE"])
            dfo_test_rmses = [r["Test_RMSE"] for r in dfo_run_records]
            dfo_test_maes = [r["Test_MAE"] for r in dfo_run_records]
            dfo_rmse_mean = float(np.mean(dfo_test_rmses))
            dfo_rmse_sd = float(np.std(dfo_test_rmses, ddof=1))
            dfo_mae_mean = float(np.mean(dfo_test_maes))
            dfo_mae_sd = float(np.std(dfo_test_maes, ddof=1))
            dfo_test_rmse = selected_run["Test_RMSE"]
            dfo_test_mae = selected_run["Test_MAE"]
            rmse_improvement = (
                (test_rmse - dfo_test_rmse) / test_rmse * 100
                if test_rmse != 0 else np.nan
            )
            mae_improvement = (
                (test_mae - dfo_test_mae) / test_mae * 100
                if test_mae != 0 else np.nan
            )

            config_row = dict(model_results[-1])
            config_row.update({
                "Population_Size": population_size,
                "Disturbance_Rate": disturbance_rate,
                "Search_Radius": DFO_SEARCH_RADIUS,
                "Parameter_Dimension": len(parameter_names),
                "Parameter_Names": str(parameter_names),
                "DFO_Selected_Run": selected_run["Run"],
                "DFO_Validation_RMSE": selected_run["Validation_RMSE"],
                "MLE_Test_RMSE": test_rmse,
                "MLE_Test_MAE": test_mae,
                "DFO_Test_RMSE": dfo_test_rmse,
                "DFO_Test_MAE": dfo_test_mae,
                "RMSE_Change_DFO_minus_MLE": dfo_test_rmse - test_rmse,
                "MAE_Change_DFO_minus_MLE": dfo_test_mae - test_mae,
                "DFO_Test_RMSE_Mean": dfo_rmse_mean,
                "DFO_Test_RMSE_SD": dfo_rmse_sd,
                "DFO_Test_MAE_Mean": dfo_mae_mean,
                "DFO_Test_MAE_SD": dfo_mae_sd,
                "DFO_RMSE_Improvement_Percent": rmse_improvement,
                "DFO_MAE_Improvement_Percent": mae_improvement,
                "DFO_Run_Count": len(dfo_run_records),
                "DFO_Objective_Evaluations_Mean": float(np.mean(
                    [r["Objective_Evaluations"] for r in dfo_run_records]
                )),
                "DFO_Iterations_Mean": float(np.mean(
                    [r["Iterations_Completed"] for r in dfo_run_records]
                ))
            })
            ticker_configuration_results.append(config_row)
            configuration_results.append(config_row)

            metric_comparison = pd.DataFrame({
                "Metric": ["RMSE", "MAE"],
                "MLE": [test_rmse, test_mae],
                "DFO (validation-best run)": [dfo_test_rmse, dfo_test_mae],
                "DFO run mean": [dfo_rmse_mean, dfo_mae_mean],
                "DFO run SD": [dfo_rmse_sd, dfo_mae_sd],
                "Improvement %": [rmse_improvement, mae_improvement]
            })
            print("\nConfiguration test comparison:")
            print(metric_comparison.to_string(
                index=False,
                formatters={
                    "MLE": "{:.6f}".format,
                    "DFO (validation-best run)": "{:.6f}".format,
                    "DFO run mean": "{:.6f}".format,
                    "DFO run SD": "{:.6f}".format,
                    "Improvement %": "{:.2f}%".format
                }
            ))
            print(
                f"Configuration summary: ticker={ticker}, "
                f"population={population_size}, disturbance={disturbance_rate:g}, "
                f"radius={DFO_SEARCH_RADIUS:g}, D={len(parameter_names)}, "
                f"runs={len(dfo_run_records)}, selected seed={selected_run['Seed']}, "
                f"mean objective evaluations={config_row['DFO_Objective_Evaluations_Mean']:.1f}, "
                f"mean iterations={config_row['DFO_Iterations_Mean']:.1f}"
            )

            forecast_column = f"DFO_N{population_size}_delta{disturbance_rate:g}"
            comparison[forecast_column] = selected_run["Forecast"]

    forecast_results[-1] = comparison


print("\n======================================")
print("BOX-JENKINS ANALYSIS COMPLETED")
print("======================================")



# Summary of baseline MLE and DFO forecasts
print("\n======================================")
print("MLE versus DFO test summary by DFO configuration")
print("======================================")
if configuration_results:
    metric_columns = [
        "Ticker", "Selected_ARIMA", "Population_Size", "Disturbance_Rate",
        "Search_Radius", "Parameter_Dimension",
        "MLE_Test_RMSE", "DFO_Test_RMSE", "DFO_Test_RMSE_Mean", "DFO_Test_RMSE_SD",
        "RMSE_Change_DFO_minus_MLE", "DFO_RMSE_Improvement_Percent",
        "MLE_Test_MAE", "DFO_Test_MAE", "DFO_Test_MAE_Mean", "DFO_Test_MAE_SD",
        "MAE_Change_DFO_minus_MLE", "DFO_MAE_Improvement_Percent",
        "DFO_Run_Count", "DFO_Selected_Run", "DFO_Objective_Evaluations_Mean",
        "DFO_Iterations_Mean"
    ]
    metric_summary = pd.DataFrame(configuration_results)
    metric_summary = metric_summary[
        [column for column in metric_columns if column in metric_summary.columns]
    ]
    display_summary = metric_summary.rename(columns={
        "Selected_ARIMA": "ARIMA",
        "Population_Size": "Population",
        "Disturbance_Rate": "Disturbance",
        "Search_Radius": "Radius",
        "Parameter_Dimension": "D",
        "MLE_Test_RMSE": "MLE RMSE",
        "DFO_Test_RMSE": "DFO RMSE (selected)",
        "DFO_Test_RMSE_Mean": "DFO RMSE mean",
        "DFO_Test_RMSE_SD": "DFO RMSE SD",
        "RMSE_Change_DFO_minus_MLE": "RMSE change",
        "DFO_RMSE_Improvement_Percent": "RMSE gain %",
        "MLE_Test_MAE": "MLE MAE",
        "DFO_Test_MAE": "DFO MAE (selected)",
        "DFO_Test_MAE_Mean": "DFO MAE mean",
        "DFO_Test_MAE_SD": "DFO MAE SD",
        "DFO_Run_Count": "Runs",
        "DFO_Selected_Run": "Selected run",
        "DFO_Objective_Evaluations_Mean": "Mean evaluations",
        "DFO_Iterations_Mean": "Mean iterations",
        "MAE_Change_DFO_minus_MLE": "MAE change",
        "DFO_MAE_Improvement_Percent": "MAE gain %"
    })

    # Render a fixed-width ASCII table, similar to Statsmodels' summary.
    headers = list(display_summary.columns)
    table_rows = []
    for _, row in display_summary.iterrows():
        cells = []
        for column in headers:
            value = row[column]
            if pd.isna(value):
                cells.append("")
            elif isinstance(value, (float, np.floating)):
                if column.endswith("%"):
                    cells.append(f"{value:.2f}%")
                else:
                    cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        table_rows.append(cells)

    widths = [
        max(len(str(header)), *(len(row[i]) for row in table_rows))
        for i, header in enumerate(headers)
    ]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    title = "MLE versus DFO test summary by configuration"
    print("\n" + "=" * len(title))
    print(title)
    print("=" * len(title))
    print(border)
    print("| " + " | ".join(
        str(header).ljust(widths[i]) for i, header in enumerate(headers)
    ) + " |")
    print(border)
    for row in table_rows:
        print("| " + " | ".join(
            row[i].rjust(widths[i]) if i > 1 else row[i].ljust(widths[i])
            for i in range(len(headers))
        ) + " |")
    print(border)

    if forecast_results:
        forecast_table = pd.concat(forecast_results)
        forecast_table.to_csv("arima_mle_dfo_forecasts_vs_actual.csv")
        metric_summary.to_csv("arima_mle_dfo_metrics_comparison.csv", index=False)
        metric_summary.to_csv(
            "arima_mle_dfo_configuration_summary.csv", index=False
        )
        pd.DataFrame(monte_carlo_results).to_csv(
            "arima_dfo_monte_carlo_runs.csv", index=False
        )
        print("\nSaved actual-versus-forecast values to "
              "arima_mle_dfo_forecasts_vs_actual.csv")
        print("Saved metric comparison to arima_mle_dfo_metrics_comparison.csv")
        print("Saved all population/disturbance results to "
              "arima_mle_dfo_configuration_summary.csv")
        print("Saved per-run Monte Carlo results to arima_dfo_monte_carlo_runs.csv")
