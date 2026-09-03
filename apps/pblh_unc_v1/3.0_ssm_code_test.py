from gruanpy.ssm.statsmodels.univariate import UnivariateLLL, UnivariateLLT
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Synthetic generators
# ---------------------------------------------------------

def generate_lll(n=500, sigma2_level=0.5, sigma2_meas=1.0, seed=0):
    rng = np.random.default_rng(seed)

    level = np.zeros(n)
    y = np.zeros(n)

    level[0] = rng.normal(0, np.sqrt(sigma2_level))

    for t in range(1, n):
        level[t] = level[t-1] + rng.normal(0, np.sqrt(sigma2_level))

    y = level + rng.normal(0, np.sqrt(sigma2_meas), size=n)

    return y, level


def generate_llt(n=500, sigma2_level=0.5, sigma2_trend=0.2, sigma2_meas=1.0, seed=0):
    rng = np.random.default_rng(seed)

    level = np.zeros(n)
    trend = np.zeros(n)
    y = np.zeros(n)

    level[0] = rng.normal(0, np.sqrt(sigma2_level))
    trend[0] = rng.normal(0, np.sqrt(sigma2_trend))

    for t in range(1, n):
        trend[t] = trend[t-1] + rng.normal(0, np.sqrt(sigma2_trend))
        level[t] = level[t-1] + trend[t-1] + rng.normal(0, np.sqrt(sigma2_level))

    y = level + rng.normal(0, np.sqrt(sigma2_meas), size=n)

    return y, level, trend


# ---------------------------------------------------------
# Test 1 — LLL with MLE measurement variance
# ---------------------------------------------------------
if False: # Test 1 & 2
    print("\n=== Test 1 - LLL (MLE measurement variance) ===")
    y, true_level = generate_lll()

    model = UnivariateLLL(y)
    res = model.fit(maxiter=200, disp=False)

    print("True params: sigma2.measurement=1.0, sigma2.level=0.5")
    print("Estimated params:", res.params)
    print(res.summary())

    smoothed_level = res.smoother_results.smoothed_state[0]

    plt.figure(figsize=(10,4))
    plt.plot(true_level, label="True level")
    plt.plot(smoothed_level, label="Smoothed level")
    plt.title("Test 1 - LLL State Recovery (MLE measurement variance)")
    plt.legend()
    plt.show()


# ---------------------------------------------------------
# Test 2 — LLL with fixed measurement variance array
# ---------------------------------------------------------

    print("\n=== Test 2 - LLL (fixed measurement variance) ===")
    meas_var = np.ones_like(y) * 1.0

    model = UnivariateLLL(y, measurement_sigma2=meas_var)
    res = model.fit(maxiter=200, disp=False)

    print("True params: sigma2.level=0.5 (measurement variance fixed to 1.0)")
    print("Estimated params:", res.params)
    print(res.summary())

    smoothed_level = res.smoother_results.smoothed_state[0]

    plt.figure(figsize=(10,4))
    plt.plot(true_level, label="True level")
    plt.plot(smoothed_level, label="Smoothed level")
    plt.title("Test 2 - LLL State Recovery (fixed measurement variance)")
    plt.legend()
    plt.show()


if True: # Test 3 & 4 & 5
    # ---------------------------------------------------------
    # Test 3 — LLT with MLE measurement variance
    # ---------------------------------------------------------
    print("\n=== Test 3 - LLT (MLE measurement variance) ===")
    y, true_level, true_trend = generate_llt()

    model = UnivariateLLT(y)
    res = model.fit(maxiter=200, disp=False)

    print("True params: sigma2.measurement=1.0, sigma2.level=0.5, sigma2.trend=0.2")
    print("Estimated params:", res.params)
    print(res.summary())

    smoothed_level = res.smoother_results.smoothed_state[0]
    smoothed_trend = res.smoother_results.smoothed_state[1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(true_level, label="True level")
    ax[0].plot(smoothed_level, label="Smoothed level")
    ax[0].set_title("Test 3 - LLT Level State Recovery")
    ax[0].legend()

    ax[1].plot(true_trend, label="True trend")
    ax[1].plot(smoothed_trend, label="Smoothed trend")
    ax[1].set_title("Test 3 - LLT Trend State Recovery")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------
    # Test 4 — LLT with fixed measurement variance array
    # ---------------------------------------------------------
    print("\n=== Test 4 - LLT (fixed measurement variance) ===")
    meas_var = np.ones_like(y) * 1.0

    model = UnivariateLLT(y, measurement_sigma2=meas_var)
    res = model.fit(maxiter=200, disp=False)

    print("True params: sigma2.level=0.5, sigma2.trend=0.2 (measurement variance fixed to 1.0)")
    print("Estimated params:", res.params)
    print(res.summary())

    smoothed_level = res.smoother_results.smoothed_state[0]
    smoothed_trend = res.smoother_results.smoothed_state[1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(true_level, label="True level")
    ax[0].plot(smoothed_level, label="Smoothed level")
    ax[0].set_title("Test 4 - LLT Level State Recovery (fixed measurement variance)")
    ax[0].legend()

    ax[1].plot(true_trend, label="True trend")
    ax[1].plot(smoothed_trend, label="Smoothed trend")
    ax[1].set_title("Test 4 - LLT Trend State Recovery (fixed measurement variance)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()


    # ---------------------------------------------------------
    # Test 5 — LLT with *inflated* fixed measurement variance
    # ---------------------------------------------------------
    print("\n=== Test 5 - LLT (inflated fixed measurement variance) ===")
    inflated_meas_var = np.ones_like(y) * 10.0

    model = UnivariateLLT(y, measurement_sigma2=inflated_meas_var)
    res = model.fit(maxiter=200, disp=False)

    print("True params: sigma2.level=0.5, sigma2.trend=0.2 (measurement variance fixed to 10.0)")
    print("Estimated params:", res.params)
    print(res.summary())

    smoothed_level = res.smoother_results.smoothed_state[0]
    smoothed_trend = res.smoother_results.smoothed_state[1]

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax[0].plot(true_level, label="True level")
    ax[0].plot(smoothed_level, label="Smoothed level")
    ax[0].set_title("Test 5 - LLT Level State Recovery (inflated measurement variance)")
    ax[0].legend()

    ax[1].plot(true_trend, label="True trend")
    ax[1].plot(smoothed_trend, label="Smoothed trend")
    ax[1].set_title("Test 5 - LLT Trend State Recovery (inflated measurement variance)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()
