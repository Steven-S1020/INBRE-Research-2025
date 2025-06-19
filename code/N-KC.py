import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    from functools import partial
    from scipy.constants import pi
    from scipy.integrate import trapezoid
    from scipy.stats import norm, cauchy, skewnorm, beta, gamma, lognorm
    from scipy.optimize import differential_evolution

    redc = "#a4031f"
    grayc = "#dddde3"
    greenc = "#88bf9b"
    bluec = "#3c91e6"
    pinkc = "#ffa0ac"

    sns.set_theme(
        style="white",
        context="talk",
        font_scale=1.25,
        rc={"axes.edgecolor": "black", "grid.color": "silver"},
    )
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams["figure.dpi"] = 115
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.labelsize"] = 20
    plt.rcParams["axes.labelpad"] = 20
    plt.rcParams["axes.titlesize"] = 26
    plt.rcParams["axes.titlepad"] = 20
    plt.rcParams["xtick.labelsize"] = 18
    plt.rcParams["ytick.labelsize"] = 18
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.minor.size"] = 6
    plt.rcParams["ytick.minor.size"] = 6
    plt.rcParams["xtick.color"] = "black"
    plt.rcParams["ytick.color"] = "black"
    epsilon = 1e-3
    return (
        FontProperties,
        beta,
        bluec,
        cauchy,
        differential_evolution,
        epsilon,
        gamma,
        grayc,
        greenc,
        lognorm,
        mo,
        norm,
        np,
        partial,
        pd,
        pi,
        pinkc,
        plt,
        redc,
        skewnorm,
        sns,
        trapezoid,
    )


@app.cell
def _(mo):
    mo.md(r"""## R: Kumaraswamy""")
    return


@app.cell
def _(np):
    def F_R(x, a, b):
        return 1 - np.power(1 - np.power(x, a), b)
    return (F_R,)


@app.cell
def _(epsilon, np):
    def f_R(x, a, b):
        base = 1.0 - np.power(x, a)
        safe_base = np.maximum(base, epsilon)
        return a * b * np.power(x, a - 1) * np.power(safe_base, b - 1)
    return (f_R,)


@app.cell
def _(mo):
    mo.md(r"""## Y: Cauchy""")
    return


@app.cell
def _(cauchy):
    def Q_Y(x, gamma):  # Quantile of Cauchy
        return cauchy.ppf(x, scale=gamma)
    return (Q_Y,)


@app.cell
def _(np, pi):
    def uQ_Y(x, gamma):  # Derivative Quantile of Cauchy
        return gamma * (1 / (np.cos(pi * (x - 0.5)) ** 2)) * pi
    return (uQ_Y,)


@app.cell
def _(mo):
    mo.md(r"""## T: Normal""")
    return


@app.cell
def _(norm):
    def F_T(x, mu):
        return norm.cdf(x, loc=mu)
    return (F_T,)


@app.cell
def _(norm):
    def f_T(x, mu):
        return norm.pdf(x, loc=mu)
    return (f_T,)


@app.cell
def _(mo):
    mo.md(r"""## Composite""")
    return


@app.cell
def _(epsilon, np):
    def n_cbl(x, mu, sig, alpha, lamb):
        f_r = (
            2 * np.arctanh(1 - 2 * lamb) * (lamb**x) * ((1 - lamb) ** (1 - x))
        ) / (1 - 2 * lamb)
        F_r = ((lamb**x) * ((1 - lamb) ** (1 - x)) + lamb - 1) / ((2 * lamb) - 1)
        s_r = 1 - F_r

        F_r = np.clip(F_r, epsilon, 1 - epsilon)
        s_r = np.clip(s_r, epsilon, 1 - epsilon)

        z = np.exp(-((alpha * np.log(F_r / (1 - F_r)) - mu) ** 2) / (2 * sig**2))

        numerator = z * alpha * (s_r + F_r) * f_r
        denominator = np.sqrt(2 * np.pi) * s_r * F_r * sig

        return numerator / denominator
    return (n_cbl,)


@app.cell
def _(F_R, Q_Y, f_R, f_T, uQ_Y):
    def n_kc(x, a, b, gamma, mu):
        return (
            f_T(Q_Y(F_R(x, a, b), gamma), mu)
            * uQ_Y(F_R(x, a, b), gamma)
            * f_R(x, a, b)
        )
    return (n_kc,)


@app.cell
def _(F_R, F_T, Q_Y):
    def N_KC(x, a, b, gamma, mu):
        return F_T(Q_Y(F_R(x, a, b), gamma), mu)
    return (N_KC,)


@app.cell
def _(mo):
    mo.md(r"""## Calculations""")
    return


@app.cell
def _(differential_evolution, epsilon, n_kc, np):
    def Tcalc_params_NKC(data, bounds):
        def Tll_nkc(params):
            a, b, gamma, mu = params
            y = n_kc(data, a, b, gamma, mu)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        result = differential_evolution(
            Tll_nkc,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-2,
            updating='immediate'
        )
        return result.x, -result.fun
    return (Tcalc_params_NKC,)


@app.cell
def _(differential_evolution, epsilon, n_kc, np, partial):
    def ll_nkc(params, data):
        a, b, gamma, mu = params
        y = n_kc(data, a, b, gamma, mu)
        y[y <= 0] = epsilon
        return -np.sum(np.log(y))

    def calc_params_NKC(data, bounds):
        result = differential_evolution(
            partial(ll_nkc, data=data),
            bounds,
            strategy="currenttobest1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-2,
            workers=-1,
            updating='deferred'
        )
        return result.x, -result.fun
    return (calc_params_NKC,)


@app.cell
def _(differential_evolution, epsilon, f_R, np):
    def calc_params_ks(data):
        def ll_ks(params):
            a, b = params
            y = f_R(data, a, b)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (1e-5, 5),   # a
            (1e-5, 5),   # b
        ]

        result = differential_evolution(
            ll_ks,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-2,
            updating="immediate",
        )
        return result.x, -result.fun
    return (calc_params_ks,)


@app.cell
def _(differential_evolution, epsilon, gamma, np):
    def calc_params_gamma(data, bounds):
        def ll_gamma(params):
            shape, scale = params
            y = gamma.pdf(data, a=shape, scale=scale)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        result = differential_evolution(
            ll_gamma,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-4,
            updating="immediate",
        )
        return result.x, -result.fun
    return


@app.cell
def _(beta, differential_evolution, epsilon, np):
    def calc_params_beta(data, bounds):
        def ll_beta(params):
            a, b = params
            y = beta.pdf(data, a, b)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        result = differential_evolution(
            ll_beta,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-8,
            updating="immediate",
        )
        return result.x, -result.fun
    return (calc_params_beta,)


@app.cell
def _(differential_evolution, epsilon, lognorm, np):
    def calc_params_lognorm(data):
        def ll_lognorm(params):
            s = params
            y = lognorm.logpdf(data, s)
            y[y <= 0] = epsilon
            return -np.sum(y)

        bounds = [
            (0.5, 5),  # s
        ]

        result = differential_evolution(
            ll_lognorm,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-4,
            updating="immediate",
        )
        return result.x, -result.fun
    return


@app.cell
def _(differential_evolution, epsilon, norm, np):
    def calc_params_norm(data):
        def ll_norm(params):
            mu, sig = params
            y = norm.pdf(data, mu, sig)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (-5, 5),  # mu
            (epsilon, 10),  # sigma
        ]

        result = differential_evolution(
            ll_norm,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-4,
            updating="immediate",
        )
        return result.x, -result.fun
    return


@app.cell
def _(differential_evolution, epsilon, n_cbl, np):
    def calc_params_ncbl(data):
        def ll_ncbl(params):
            mu, sig, alpha, lamb = params
            y = n_cbl(data, mu, sig, alpha, lamb)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (-5, 5),  # mu
            (epsilon, 5),  # sigma
            (epsilon, 5),  # alpha
            (epsilon, 1 - epsilon),  # lambda
        ]

        result = differential_evolution(
            ll_ncbl,
            bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-4,
            updating="immediate",
        )
        return result.x, -result.fun
    return (calc_params_ncbl,)


@app.function
def AIC(k, ll, precision=None):
    # k    : Number of Parameters
    # ll   : Log-Likelihood Value
    if precision is None:
        return (2 * k) - (2 * ll)
    else:
        return round((2 * k) - (2 * ll), precision)


@app.cell
def _(np):
    def BIC(k, n, ll, precision=None):
        # k    : Number of Parameters
        # n    : Number of Data Points
        # ll   : Log-Likelihood Value
        if precision is None:
            return (k * np.log(n)) - (2 * ll)
        else:
            return round(((k * np.log(n)) - (2 * ll)), precision)
    return (BIC,)


@app.cell
def _(mo):
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _(epsilon):
    def ScaleAdjustData(raw):
        if raw.min() < 0:
            adjusted = raw + abs(raw.min() + epsilon)
        else:
            adjusted = raw - raw.min() + epsilon
        return adjusted / (adjusted.max() + epsilon)
    return (ScaleAdjustData,)


@app.cell
def _():
    nkc_param_names = ["a", "b", "\\lambda", "\\mu"]
    ncbl_param_names = ["\\mu", "\\sigma", "\\alpha", "\\lambda"]
    beta_param_names = ["\\alpha", "\\beta"]
    norm_param_names = ['\\mu', '\\sigma']
    gamma_param_names = ['\\alpha', '\\theta']
    ks_param_names = ['a', 'b']



    def format_label(
        params,
        label="",
        precision=2,
        param_width=10,
        sci_threshold=1e4,
        small_threshold=0.1,
    ):
        """
        Create a one-line label string with fixed-width aligned param blocks.

        Parameters:
            params       : dict of {name: value}
            label        : string prefix (e.g., 'norm1')
            precision    : decimal places for float values
            param_width  : total width of each 'key=value' string
                          (spaces are added with .ljust())
        """
        if not isinstance(params, dict):
            raise TypeError("params must be a dict")

        param_strs = []
        for k, v in params.items():
            if abs(v) >= sci_threshold or (abs(v) < small_threshold and v != 0):
                value_str = f"{v:.{precision}e}"
                raw = f"${k}={value_str}$"
                n = (
                    (param_width - len(raw) - 1)
                    if "\\" not in k
                    else (param_width - (len(raw) - len(k)) - 1)
                )
                padded = raw + (" " * max(0, n))
            else:
                value_str = f"{v:.{precision}f}"
                raw = f"${k}={value_str}$"
                n = (
                    (param_width - len(raw))
                    if "\\" not in k
                    else (param_width - (len(raw) - len(k)))
                )
                padded = raw + (" " * max(0, n))

            param_strs.append(padded)
        return f"{label}  " + "".join(param_strs)
    return (
        beta_param_names,
        format_label,
        ks_param_names,
        ncbl_param_names,
        nkc_param_names,
    )


@app.cell
def _(mo):
    mo.md(r"""## PDF & CDF of $\,N\!-\!K\{C\}\,$""")
    return


@app.cell
def _(N_KC, Tcalc_params_NKC, epsilon, n_kc, norm, np, skewnorm):
    x = np.linspace(epsilon, 1 - epsilon, 10000)

    norm1 = np.clip(
        norm.rvs(loc=0.5, scale=0.09, size=10000), epsilon, 1 - epsilon
    )
    norm2 = np.clip(
        norm.rvs(loc=0.5, scale=0.15, size=10000), epsilon, 1 - epsilon
    )
    rskew = np.clip(
        skewnorm.rvs(a=25, loc=0.1, scale=0.25, size=10000), epsilon, 1 - epsilon
    )
    lskew = np.clip(
        skewnorm.rvs(a=-25, loc=0.9, scale=0.25, size=10000), epsilon, 1 - epsilon
    )
    bimod = np.clip(
        np.concatenate(
            [
                norm.rvs(loc=0.3, scale=0.065, size=5000),
                norm.rvs(loc=0.7, scale=0.065, size=5000),
            ]
        ),
        epsilon,
        1 - epsilon,
    )

    bounds = [
        (1e-5, 2.5),  # a
        (1e-5, 2.5),  # b
        (1e-5, 20),   # gamma
        (-30, 30),     # mu
    ]

    norm1_params, norm1_ll = Tcalc_params_NKC(norm1, bounds)
    norm2_params, norm2_ll = Tcalc_params_NKC(norm2, bounds)
    rskew_params, rskew_ll = Tcalc_params_NKC(rskew, bounds)
    lskew_params, lskew_ll = Tcalc_params_NKC(lskew, bounds)
    bimod_params, bimod_ll = Tcalc_params_NKC(bimod, bounds)

    nkc_norm1 = n_kc(x, *norm1_params)
    nkc_norm2 = n_kc(x, *norm2_params)
    nkc_rskew = n_kc(x, *rskew_params)
    nkc_lskew = n_kc(x, *lskew_params)
    nkc_bimod = n_kc(x, *bimod_params)

    NKC_norm1 = N_KC(x, *norm1_params)
    NKC_norm2 = N_KC(x, *norm2_params)
    NKC_rskew = N_KC(x, *rskew_params)
    NKC_lskew = N_KC(x, *lskew_params)
    NKC_bimod = N_KC(x, *bimod_params)
    return (
        NKC_bimod,
        NKC_lskew,
        NKC_norm1,
        NKC_norm2,
        NKC_rskew,
        bimod_params,
        bounds,
        lskew_params,
        nkc_bimod,
        nkc_lskew,
        nkc_norm1,
        nkc_norm2,
        nkc_rskew,
        norm1_params,
        norm2_params,
        rskew_params,
        x,
    )


@app.cell
def _(mo):
    mo_a = mo.ui.number(start=0.001, stop=2.5, step=0.00001, value=0.0011, label=rf"$a$")
    mo_b = mo.ui.number(start=0.00000001, stop=2.5, step=0.00001, value=0.000755, label=rf"$b$")
    mo_gamma = mo.ui.number(start=0.1, stop=30, step=0.1, value=1.21, label=rf"$\gamma$")
    mo_mu = mo.ui.number(start=-80, stop=30, step=0.001, value=-71.644, label=rf"$\mu$")
    mo_sigma = mo.ui.number(start=0.001, stop=30, step=0.001, value=3.014, label=rf"$\sigma$")

    mo.hstack([mo_a, mo_b, mo_gamma, mo_mu, mo_sigma], justify="center", align="center", gap=5,)
    return


@app.cell
def _(
    FontProperties,
    NKC_bimod,
    NKC_lskew,
    NKC_norm1,
    NKC_norm2,
    NKC_rskew,
    bimod_params,
    bluec,
    format_label,
    grayc,
    greenc,
    lskew_params,
    mo,
    nkc_bimod,
    nkc_lskew,
    nkc_norm1,
    nkc_norm2,
    nkc_param_names,
    nkc_rskew,
    norm1_params,
    norm2_params,
    pinkc,
    plt,
    redc,
    rskew_params,
    x,
):
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

    label_norm1 = format_label(dict(zip(nkc_param_names, norm1_params)), label="norm1", param_width=15)
    label_norm2 = format_label(dict(zip(nkc_param_names, norm2_params)), label="norm2", param_width=15)
    label_rskew = format_label(dict(zip(nkc_param_names, rskew_params)), label="rskew", param_width=15)
    label_lskew = format_label(dict(zip(nkc_param_names, lskew_params)), label="lskew", param_width=15)
    label_bimod = format_label(dict(zip(nkc_param_names, bimod_params)), label="bimod", param_width=15)

    ax1.plot(x, nkc_norm1, label=label_norm1, color=redc, lw=2)
    ax1.plot(x, nkc_norm2, label=label_norm2, color=grayc, lw=2)
    ax1.plot(x, nkc_rskew, label=label_rskew, color=greenc, lw=2)
    ax1.plot(x, nkc_lskew, label=label_lskew, color=bluec, lw=2)
    ax1.plot(x, nkc_bimod, label=label_bimod, color=pinkc, lw=2)
    #ax1.plot(x, n_kc(x, mo_a.value, mo_b.value, mo_gamma.value, mo_mu.value))

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5.6)
    ax1.set_xlabel(xlabel=r"$X$ Value")
    ax1.set_ylabel("Density")

    ax1.set_title(r'$\,N\!-\!K\{C\}\,$ PDF')
    ax1.legend(prop=FontProperties(family="monospace", size=10))
    ax1.grid()

    ax2.plot(x, NKC_norm1, label="norm1", color=redc, lw=2)
    ax2.plot(x, NKC_norm2, label="norm2", color=grayc, lw=2)
    ax2.plot(x, NKC_rskew, label="rskew", color=greenc, lw=2)
    ax2.plot(x, NKC_lskew, label="lskew", color=bluec, lw=2)
    ax2.plot(x, NKC_bimod, label="bimod", color=pinkc, lw=2)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$X$ Value")
    ax2.set_ylabel("Probability")

    ax2.set_title(r'$\,N\!-\!K\{C\}\,$ CDF')
    ax2.legend(prop=FontProperties(family="monospace", size=10))
    ax2.grid()

    mo.as_html(fig1.gca()).center()
    return


@app.cell
def _(nkc_bimod, nkc_lskew, nkc_norm1, nkc_norm2, nkc_rskew, trapezoid, x):
    print("Area under nkc_norm1:", trapezoid(nkc_norm1, x))
    print("Area under nkc_norm2:", trapezoid(nkc_norm2, x))
    print("Area under nkc_lskew:", trapezoid(nkc_lskew, x))
    print("Area under nkc_rskew:", trapezoid(nkc_rskew, x))
    print("Area under nkc_bimod:", trapezoid(nkc_bimod, x))
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Fitting Example One""")
    return


@app.cell
def _(pd):
    df_GAQ = pd.read_csv('./data/Global Air Quality (2024) - 6 Cities/New_York_Air_Quality.csv', usecols=['NO2'])
    return (df_GAQ,)


@app.cell
def _(
    ScaleAdjustData,
    Tcalc_params_NKC,
    bounds,
    calc_params_beta,
    calc_params_ks,
    calc_params_ncbl,
    df_GAQ,
):
    raw_NO2 = df_GAQ["NO2"].values
    scaled_NO2 = ScaleAdjustData(raw_NO2)

    bounds_beta = [
        (1, 5),  # a
        (1, 5),  # b
    ]

    nkc_ny_no2_params, nkc_ny_no2_ll = Tcalc_params_NKC(scaled_NO2, bounds)
    beta_ny_no2_params, beta_ny_no2_ll = calc_params_beta(scaled_NO2, bounds_beta)
    ncbl_ny_no2_params, ncbl_ny_no2_ll = calc_params_ncbl(scaled_NO2)
    ks_ny_no2_params, ks_ny_no2_ll = calc_params_ks(scaled_NO2)
    return (
        beta_ny_no2_ll,
        beta_ny_no2_params,
        bounds_beta,
        ks_ny_no2_ll,
        ks_ny_no2_params,
        ncbl_ny_no2_ll,
        ncbl_ny_no2_params,
        nkc_ny_no2_ll,
        nkc_ny_no2_params,
        raw_NO2,
        scaled_NO2,
    )


@app.cell
def _(
    BIC,
    FontProperties,
    beta,
    beta_ny_no2_ll,
    beta_ny_no2_params,
    beta_param_names,
    bluec,
    f_R,
    format_label,
    grayc,
    greenc,
    ks_ny_no2_ll,
    ks_ny_no2_params,
    ks_param_names,
    mo,
    n_cbl,
    n_kc,
    ncbl_ny_no2_ll,
    ncbl_ny_no2_params,
    ncbl_param_names,
    nkc_ny_no2_ll,
    nkc_ny_no2_params,
    nkc_param_names,
    pd,
    pinkc,
    plt,
    raw_NO2,
    redc,
    scaled_NO2,
    sns,
    x,
):
    fig3, ax3 = plt.subplots(dpi=100)
    sns.histplot(scaled_NO2, bins=70, stat="density", ax=ax3, color=grayc)

    label_nkc = format_label(dict(zip(nkc_param_names, nkc_ny_no2_params)), label=r"$\,N\!-\!K\{C\}\ $ ")
    label_beta = format_label(dict(zip(beta_param_names, beta_ny_no2_params)), label=r"Beta     ")
    label_ncbl = format_label(dict(zip(ncbl_param_names, ncbl_ny_no2_params)), label=r"$\,N\!-\!CB\{L\}\ $",)
    label_ks = format_label(dict(zip(ks_param_names, ks_ny_no2_params)), label='KS       ')

    ax3.plot(x, n_kc(x, *nkc_ny_no2_params), label=label_nkc, color=redc, lw=2)
    ax3.plot(x, beta.pdf(x, *beta_ny_no2_params), label=label_beta, color=bluec, lw=2, ls="--",)
    ax3.plot(x, n_cbl(x, *ncbl_ny_no2_params), label=label_ncbl, color=greenc, lw=2, ls="--",)
    ax3.plot(x, f_R(x, *ks_ny_no2_params), label=label_ks, color=pinkc, lw=2, ls='--')

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 3.6)
    ax3.set_xlabel(r'NO2 in µg/m³ scaled by a factor of $1\colon99.1$')
    ax3.set_ylabel("Density")

    ax3.set_title("Nitrogen Dioxide Concentration in µg/m³")
    ax3.legend(prop=FontProperties(family="monospace", size=12))
    ax3.grid()

    result_data = {
        "Distributions": ['NKC', 'NCBL', 'Beta', 'KS'],
        "AIC": [
            AIC(4, nkc_ny_no2_ll, precision=2),
            AIC(4, ncbl_ny_no2_ll, precision=2),
            AIC(2, beta_ny_no2_ll, precision=2),
            AIC(2, ks_ny_no2_ll, precision=2),
        ],
        "BIC": [
            BIC(4, raw_NO2.size, nkc_ny_no2_ll, precision=2),
            BIC(4, raw_NO2.size, ncbl_ny_no2_ll, precision=2),
            BIC(2, raw_NO2.size, beta_ny_no2_ll, precision=2),
            BIC(2, raw_NO2.size, ks_ny_no2_ll, precision=2)
        ],
    }
    df_res = pd.DataFrame(result_data)

    mo.hstack([mo.as_html(fig3), mo.as_html(df_res).style(width="800px")], justify='center', gap=5)
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Fitting Example Two""")
    return


@app.cell
def _(pd):
    df_HA = pd.read_csv('./data/HeartAttacks/Medicaldataset.csv')
    return (df_HA,)


@app.cell
def _(ScaleAdjustData, df_HA):
    raw_HR = df_HA.loc[(df_HA['Result'] == 'positive') & (df_HA['Heart rate'] < 200), 'Heart rate']
    scaled_HR = ScaleAdjustData(raw_HR)
    return raw_HR, scaled_HR


@app.cell
def _(
    bounds,
    bounds_beta,
    calc_params_NKC,
    calc_params_beta,
    calc_params_ks,
    calc_params_ncbl,
    epsilon,
    scaled_HR,
):
    bounds_gamma_test = [
        (epsilon, 200),  # a
        (epsilon, 20),  # b
    ]

    nkc_HR_params, nkc_HR_ll = calc_params_NKC(scaled_HR, bounds) 
    beta_HR_params, beta_HR_ll = calc_params_beta(scaled_HR, bounds_beta)
    ncbl_HR_params, ncbl_HR_ll = calc_params_ncbl(scaled_HR)
    ks_HR_params, ks_HR_ll = calc_params_ks(scaled_HR)
    return (
        beta_HR_ll,
        beta_HR_params,
        ks_HR_ll,
        ks_HR_params,
        ncbl_HR_ll,
        ncbl_HR_params,
        nkc_HR_ll,
        nkc_HR_params,
    )


@app.cell
def _(
    BIC,
    FontProperties,
    beta,
    beta_HR_ll,
    beta_HR_params,
    beta_param_names,
    bluec,
    f_R,
    format_label,
    grayc,
    greenc,
    ks_HR_ll,
    ks_HR_params,
    ks_param_names,
    mo,
    n_cbl,
    n_kc,
    ncbl_HR_ll,
    ncbl_HR_params,
    ncbl_ny_no2_params,
    ncbl_param_names,
    nkc_HR_ll,
    nkc_HR_params,
    nkc_param_names,
    pd,
    pinkc,
    plt,
    raw_HR,
    redc,
    scaled_HR,
    sns,
    x,
):
    fig4, ax4 = plt.subplots(dpi=100)
    sns.histplot(scaled_HR, bins=25, stat="density", ax=ax4, color=grayc)

    label_nkc_HR = format_label(dict(zip(nkc_param_names, nkc_HR_params)), label=r'$\,N\!-\!K\{C\}\ $  ')
    label_beta_HR = format_label(dict(zip(beta_param_names, beta_HR_params)), label=r'Beta      ')
    label_ncbl_HR = format_label(dict(zip(ncbl_param_names, ncbl_ny_no2_params)), label=r"$\,N\!-\!CB\{L\}\ $ ",)
    label_ks_HR = format_label(dict(zip(ks_param_names, ks_HR_params)), label='KS        ')

    ax4.plot(x, n_kc(x, *nkc_HR_params), label=label_nkc_HR, color=redc, lw=2)
    ax4.plot(x, beta.pdf(x, *beta_HR_params), label=label_beta_HR, color=bluec, lw=2, ls='--')
    ax4.plot(x, n_cbl(x, *ncbl_HR_params), label=label_ncbl_HR, color=greenc, lw=2, ls='--')
    ax4.plot(x, f_R(x, *ks_HR_params), label=label_ks_HR, color=pinkc, lw=2, ls='--')

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 4.6)
    ax4.set_xlabel(r'Heart Rate BPM Scaled by factor of  $1\colon135$')
    ax4.set_ylabel("Density")

    ax4.set_title('Avg. Heart Rate for People who had Heart Attacks in Iraq')
    ax4.legend(prop=FontProperties(family="monospace", size=12))
    ax4.grid()

    result_HR_data = {
        "Distributions": ['NKC', 'Beta', 'NCBL', 'KS'],
        "AIC": [
            AIC(4, nkc_HR_ll, precision=2),
            AIC(2, beta_HR_ll, precision=2),
            AIC(4, ncbl_HR_ll, precision=2),
            AIC(2, ks_HR_ll, precision=2),
        ],
        "BIC": [
            BIC(4, raw_HR.size, nkc_HR_ll, precision=2),
            BIC(2, raw_HR.size, beta_HR_ll, precision=2),
            BIC(4, raw_HR.size, ncbl_HR_ll, precision=2),
            BIC(2, raw_HR.size, ks_HR_ll, precision=2),
        ],
    }
    df_HR_res = pd.DataFrame(result_HR_data)

    mo.hstack([mo.as_html(fig4), mo.as_html(df_HR_res).style(width="800px")], justify='center', gap=5)
    return


if __name__ == "__main__":
    app.run()
