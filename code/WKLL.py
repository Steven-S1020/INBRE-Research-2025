import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd
    from math import sqrt
    from scipy import special
    from scipy.special import erf, gammainc, gamma
    from scipy.constants import pi
    from scipy.stats import gamma, norm, cauchy, expon, skewnorm, weibull_min
    from scipy.integrate import quad
    from scipy.optimize import minimize, differential_evolution
    from scipy.integrate import trapezoid, simpson
    sns.set_theme(style='white', context='talk', font_scale=1, palette='pastel')

    plt.rcParams['figure.figsize']=(10,8)
    plt.rcParams['figure.dpi'] = 85

    epsilon = 1e-16


    #Thank you Steven Stokes!
    return (
        differential_evolution,
        epsilon,
        mo,
        norm,
        np,
        pd,
        plt,
        simpson,
        skewnorm,
        sns,
        trapezoid,
    )


@app.cell
def _(mo):
    mo.md(f"""Kumaraswamy""")
    return


@app.cell
def _(epsilon, np):
    def FR(x, a, b):  # Safe Gamma-like CDF
        result = 1 - np.power(1 - np.power(x, a), b)
        result[result <= 0] = epsilon
        result[result >= 1] = 1 - epsilon
        return result
    return (FR,)


@app.cell
def _(epsilon, np):
    def fR(x, a, b):  # Safe Gamma-like PDF
        base = 1.0 - np.power(x, a)
        safe_base = np.maximum(base, epsilon)
        return a * b * np.power(x, a - 1) * np.power(safe_base, b - 1)
    return (fR,)


@app.cell
def _(mo):
    mo.md(r"""Log-Logistic""")
    return


@app.cell
def _(np):
    def QY(x, alpha, beta):  # Quantile of the log-logistic
        z = 1 / beta
        w = x / (1 - x)
        return alpha * np.power(w,z)

    return (QY,)


@app.cell
def _(np):
    def dQY(x, alpha, beta):
        num = alpha * np.power((x / (1 - x)), (1 / beta))
        den = beta * (x - 1) * x
        return -(num / den)
    return (dQY,)


@app.cell
def _(np):
    def fY(x, alpha, beta):
        # Convert x to a NumPy array for element-wise operations
        x = np.asarray(x, dtype=np.float64)

        # Avoid division by zero or very small values
        eps = 1e-12
        alpha = max(alpha, eps)          # ensure alpha is not zero

        # Compute safely
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            base = x / alpha
            power_term = np.power(base, beta - 1)
            denominator_term = np.power(1 + np.power(base, beta), 2)
            numerator = (beta / alpha) * power_term
            result = numerator / denominator_term

            # Replace NaNs or infs with 0s (if x was too small/large)
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return result

    return


@app.cell
def _(mo):
    mo.md(r"""Weibull Child""")
    return


@app.cell
def _(np):
    def FT(x, k, lam):
        return 1 - np.exp(-np.power((x / lam), k))
    return (FT,)


@app.cell
def _(np):
    def fT(x, k, lam):  # PDF of the Weibull distribution
        z = np.power((x / lam), (k - 1))
        w = np.power((x / lam), k)
        v = k / lam
        return (k / lam) * z * np.exp(-w)
    return (fT,)


@app.cell
def _(mo):
    mo.md(r"""CDF Creation""")
    return


@app.cell
def _(FR, FT, QY):
    def GX(x, alpha, beta, k, lam, xmin, a):
        CDF = FT(QY(FR(x, alpha, beta), xmin, a), k, lam)
        return CDF
    return (GX,)


@app.cell
def _(mo):
    mo.md(r"""PDF Creation""")
    return


@app.cell
def _(FR, QY, dQY, fR, fT):
    def gX(x, a, b, alpha, beta, k, lam):
        A = fT(QY(FR(x, a, b), alpha, beta), k, lam)
        b1 = FR(x, a, b)
        B = dQY(b1, alpha, beta)
        C = fR(x, a, b)
        return A * B * C
    return (gX,)


@app.cell
def _(mo):
    a = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    b = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    alpha = mo.ui.number(start=0.1, stop=100, step=0.1, value=1)
    beta = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    k = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    lam = mo.ui.number(start=0.1, stop=100, step=0.05, value=0.2)

    mo.hstack([a, b, alpha, beta, k, lam],justify='center',align='center',gap=5)

    #18.08661009 45.40967728 18.32785408  0.12854204 11.86199863 39.90450092
    return a, alpha, b, beta, k, lam


@app.cell
def _(a, alpha, b, beta, gX, k, lam, np, plt):


    x1 = np.linspace(1e-12, 1 - 1e-12, 100000)

    y1 = gX(x1, a.value, b.value, alpha.value, beta.value, k.value, lam.value)

    fig, ax = plt.subplots()
    ax.plot(x1, y1, label="gX(x)")
    ax.set_title(f"a={a.value}, b={b.value}, //alpha={alpha.value}, //beta={beta.value}, k={k.value}, lam={lam.value}")
    ax.set_xlabel("x")
    ax.set_ylabel("gX(x)")
    ax.grid(True)
    ax.legend()
    return x1, y1


@app.cell
def _(GX, a, alpha, b, beta, k, lam, np, plt, x1):
    xcdf = np.linspace(1e-12, 1 - 1e-12, 100000)

    ycdf = GX(x1, a.value, b.value, alpha.value, beta.value, k.value, lam.value)

    fig6, ax6 = plt.subplots()
    ax6.plot(xcdf, ycdf, label="gX(x)")
    ax6.set_title(f"α={alpha.value}, β={beta.value}, a={a.value}, b={b.value}, k={k.value}, lam={lam.value}")
    ax6.set_xlabel("x")
    ax6.set_ylabel("gX(x)")
    ax6.grid(True)
    ax6.legend()
    return


@app.cell
def _(mo):
    mo.md(r"""Notes for the future -- keep beta at 1, alpha at 3""")
    return


@app.cell
def _(x1, y1):
    print("x1 shape:", x1.shape)
    print("y1 shape:", y1.shape)

    return


@app.cell
def _(mo):
    mo.md(r"""Parameter Optimization""")
    return


@app.cell
def _(differential_evolution, gX, np):
    def calc_params_WKLL(data):
        # Log-likelihood function with zero protection
        def ll_WKLL(params):
            a, b, alpha, beta, k, lam = params
            y = gX(data, a, b, alpha, beta, k, lam)
            y[y <= 0] = 1e-12
            return -np.sum(np.log(y))

        # Bounds (tune if needed)
        bounds = [
            (1e-5, 10),    # a
            (1e-5, 10),    # b
            (1e-5, 50),    # alpha
            (2.1, 50),    # beta
            (2.1, 30),    # k
            (1e-5, 30)     # lam
        ]

        # Run global optimization
        result = differential_evolution(
            ll_WKLL,
            bounds,
            strategy="best1bin",     # Good default mutation strategy
            maxiter=1000,            # 🚀 Increase number of generations (default is 100)
            popsize=20,              # 🚀 Increase number of individuals per generation (default is 15)
            mutation=(0.5, 1.0),     # Mutation strength (can widen range to explore more)
            recombination=0.7,       # Crossover probability (how much mixing between solutions)
            polish=True,             # Fine-tune best solution with L-BFGS-B at the end
            tol=1e-4,                # Convergence tolerance (lower = more precise)
            updating="immediate",   # Update as soon as better solution is found (more aggressive)
            )

        print("Best log-likelihood:", -result.fun)
        print("Best parameters:", result.x)
        return result.x    
        # From Steven Stokes
    return (calc_params_WKLL,)


@app.cell
def _(np):
    x = np.linspace(1e-12,1 - 1e-12, 10000)
    return (x,)


@app.cell
def _(mo):
    mo.md(r"""Calc Params Code""")
    return


@app.cell
def _(calc_params_WKLL, norm, np, skewnorm):
    norm1 = np.clip(norm.rvs(loc=0.5, scale=0.09, size=10000), 1e-5, 1 - 1e-5)
    norm2 = np.clip(norm.rvs(loc=0.5, scale=0.15, size=10000), 1e-12, 1 - 1e-12)
    rskew = np.clip(skewnorm.rvs(a=25, loc=0.1, scale=0.25, size=10000), 1e-12, 1 - 1e-12)
    lskew = np.clip(skewnorm.rvs(a=-25, loc=0.9, scale=0.25, size=10000), 1e-12, 1 - 1e-12)
    bimod = np.clip(np.concatenate([norm.rvs(loc=0.3, scale=0.065, size=5000), norm.rvs(loc=0.7, scale=0.065, size=5000)]), 1e-12, 1 - 1e-12)
    #lexp = np.clip(
    #    0.8 + np.where(np.random.rand(1000) < 0.5, -expon.rvs(scale=0.1, size=1000), expon.rvs(scale=0.1, size=1000)),
    #    1e-12, 1 - 1e-12
    #)
    #rexp = np.clip(expon.rvs(loc=0.5, scale=0.2, size=1000), 1e-12, 1 - 1e-12)


    #norm1_params = calc_params_WKLL(norm1)
    #norm2_params = calc_params_WKLL(norm2)
    rskew_params = calc_params_WKLL(rskew)
    #lskew_params = calc_params_WKLL(lskew)
    bimod_params = calc_params_WKLL(bimod)
    #lexp_params = calc_params_WKLL(lexp)
    #rexp_params = calc_params_WKLL(rexp)
    return rskew, rskew_params


@app.cell
def _(gX, plt, rskew, rskew_params, sns, x):
    fig5, ax5 = plt.subplots()
    sns.histplot(rskew, bins=70, stat='density', ax=ax5, color='gray')
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 5)
    #ax5.plot(x, fT(x, 1.9, 0.5))
    ax5.plot(x, gX(x, *rskew_params), lw=3, color='red')
    ax5.grid()
    fig5.gca()
    return


@app.cell
def _(gX, plt, x):
    fig4, ax4 = plt.subplots()
    ax4.plot(x, gX(x, 16.935, 99.911, 71.027, 17.792, 5.912, 51.455), lw=3)
    ax4.plot(x, gX(x, 1.1, 3.0, 1.1, 1.6, 4.6, 2.0), lw=3)
    ax4.plot(x, gX(x, 16.400, 79.690, 96.318, 15.426, 7.382, 92.802), lw=3)
    ax4.plot(x, gX(x, 1.1, 2.1, 0.9, 0.9, 0.9, 0.2), lw=3)
    ax4.plot(x, gX(x, 6.3, 1.9, 1.1, 0.7, 0.9, 1.9), lw=3)
    ax4.plot(x, gX(x, 3.5, 2.4, 1.3, 1,  0.3, 0.85), lw=3)

    return


@app.cell
def _(np):
    xcheck = np.linspace(1e-12, 0.1, 1000000)
    return


@app.cell
def _(gX, simpson, trapezoid, x):
    print('Area under Normal:', trapezoid(gX(x, 16.935, 99.911, 71.027, 17.792, 5.912, 51.455), x))

    print('Area under Left Skew:', trapezoid(gX(x, 1.1, 3.0, 1.1, 1.6, 4.6, 2.0), x))

    print('Area under Right Skew:', trapezoid(gX(x, 16.400, 79.690, 96.318, 15.426, 7.382, 92.802), x))

    print('Area under Bimodal Skew:', trapezoid(gX(x, 9.19603514, 55.46962692, 53.96927219, 37.25840012,  7.67088249, 54.30550725), x))

    print('Area under Bimodal Skew:', simpson(gX(x, 3.5, 2.4, 1.3, 1,  0.3, 0.85), x))
    return


@app.cell
def _(mo):
    mo.md(r"""### Data Application""")
    return


@app.cell
def _(pd):
    df = pd.read_csv("./data/London_Air_Quality.csv")
    data = df["NO2"].values
    return (data,)


@app.cell
def _(data, epsilon):
    adjData = data + epsilon + abs(data.min())
    COscaled = adjData / ( adjData.max() + epsilon )
    return (COscaled,)


@app.cell
def _(COscaled):
    COscaled
    return


@app.cell
def _(COscaled, calc_params_WKLL):
    app_params = calc_params_WKLL(COscaled)
    return (app_params,)


@app.cell
def _(COscaled, app_params, gX, plt, sns, x):
    fig7, ax7 = plt.subplots()
    sns.histplot(COscaled, bins=50, ax=ax7)

    ax7.plot(x, gX(x, *app_params), lw=2)
    return (ax7,)


@app.cell
def _(COscaled, app_params, ax7, gX, plt, sns, x):
    fig8, ax8 = plt.subplots()
    sns.histplot(COscaled, bins=46, ax=ax7)
    ax8.plot(x, gX(x, *app_params), lw=2)
    return


@app.cell
def _(mo):
    mo.md(r"""75.64729711 15.66737991 32.36835104  1.0035376  33.37492693 48.338096""")
    return


if __name__ == "__main__":
    app.run()
