import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import jinja2
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties

    from scipy.constants import pi
    from scipy.integrate import trapezoid
    from scipy.stats import norm, cauchy, skewnorm, beta, gamma
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
    epsilon = 1e-16
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
        mo,
        norm,
        np,
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
    def F_T(x, mu, sigma):
        return norm.cdf(x, loc=mu, scale=sigma)
    return (F_T,)


@app.cell
def _(norm):
    def f_T(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)
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
    def n_kc(x, a, b, gamma, mu, sigma):
        return (
            f_T(Q_Y(F_R(x, a, b), gamma), mu, sigma)
            * uQ_Y(F_R(x, a, b), gamma)
            * f_R(x, a, b)
        )
    return (n_kc,)


@app.cell
def _(F_R, F_T, Q_Y):
    def N_KC(x, a, b, gamma, mu, sigma):
        return F_T(Q_Y(F_R(x, a, b), gamma), mu, sigma)
    return (N_KC,)


@app.cell
def _(mo):
    mo.md(r"""## Calculations""")
    return


@app.cell
def _(differential_evolution, epsilon, n_kc, np):
    def calc_params_NKC(data):
        def ll_nkc(params):
            a, b, gamma, mu, sigma = params
            y = n_kc(data, a, b, gamma, mu, sigma)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (1e-12, 2.5),  # a
            (1e-12, 2.5),  # b
            (1e-12, 30),  # lam
            (-30, 30),  # mu
            (1e-12, 30),  # sigma
        ]

        result = differential_evolution(
            ll_nkc,
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
    return (calc_params_NKC,)


@app.cell
def _(differential_evolution, epsilon, gamma, np):
    def calc_params_gamma(data):
        def ll_gamma(params):
            shape, scale = params
            y = gamma.pdf(data, a=shape, scale=scale)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (1e-5, 100),  # shape
            (1e-5, 100),  # scale
        ]

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
    def calc_params_beta(data):
        def ll_beta(params):
            a, b = params
            y = beta.pdf(data, a, b)
            y[y <= 0] = epsilon
            return -np.sum(np.log(y))

        bounds = [
            (1, 5),  # a
            (1, 5),  # b
        ]

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
def _():
    nkc_param_names = ["a", "b", "\\lambda", "\\mu", "\\sigma"]
    ncbl_param_names = ["\\mu", "\\sigma", "\\alpha", "\\lambda"]
    beta_param_names = ["\\alpha", "\\beta"]


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
    return beta_param_names, format_label, ncbl_param_names, nkc_param_names


@app.cell
def _(mo):
    mo.md(r"""## PDF & CDF of $\,N\!-\!K\{C\}\,$""")
    return


@app.cell
def _(calc_params_NKC, epsilon, n_kc, norm, np, skewnorm):
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

    norm1_params, norm1_ll = calc_params_NKC(norm1)
    norm2_params, norm2_ll = calc_params_NKC(norm2)
    rskew_params, rskew_ll = calc_params_NKC(rskew)
    lskew_params, lskew_ll = calc_params_NKC(lskew)
    bimod_params, bimod_ll = calc_params_NKC(bimod)

    nkc_norm1 = n_kc(x, *norm1_params)
    nkc_norm2 = n_kc(x, *norm2_params)
    nkc_rskew = n_kc(x, *rskew_params)
    nkc_lskew = n_kc(x, *lskew_params)
    nkc_bimod = n_kc(x, *bimod_params)
    return (
        bimod_params,
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
    mo_a = mo.ui.number(
        start=0.001, stop=2.5, step=0.00001, value=0.0011, label=rf"$a$"
    )
    mo_b = mo.ui.number(
        start=0.00000001, stop=2.5, step=0.00001, value=0.000755, label=rf"$b$"
    )
    mo_gamma = mo.ui.number(
        start=0.1, stop=30, step=0.1, value=1.21, label=rf"$\gamma$"
    )
    mo_mu = mo.ui.number(
        start=-80, stop=30, step=0.001, value=-71.644, label=rf"$\mu$"
    )
    mo_sigma = mo.ui.number(
        start=0.001, stop=30, step=0.001, value=3.014, label=rf"$\sigma$"
    )

    mo.hstack(
        [mo_a, mo_b, mo_gamma, mo_mu, mo_sigma],
        justify="center",
        align="center",
        gap=5,
    )
    return


@app.cell
def _(
    FontProperties,
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
    fig1, ax1 = plt.subplots()

    label_norm1 = format_label(
        dict(zip(nkc_param_names, norm1_params)), label="norm1", param_width=15
    )
    label_norm2 = format_label(
        dict(zip(nkc_param_names, norm2_params)), label="norm2", param_width=15
    )
    label_rskew = format_label(
        dict(zip(nkc_param_names, rskew_params)), label="rskew", param_width=15
    )
    label_lskew = format_label(
        dict(zip(nkc_param_names, lskew_params)), label="lskew", param_width=15
    )
    label_bimod = format_label(
        dict(zip(nkc_param_names, bimod_params)), label="bimod", param_width=15
    )

    ax1.plot(x, nkc_norm1, label=label_norm1, color=redc, lw=2)
    ax1.plot(x, nkc_norm2, label=label_norm2, color=grayc, lw=2)
    ax1.plot(x, nkc_rskew, label=label_rskew, color=greenc, lw=2)
    ax1.plot(x, nkc_lskew, label=label_lskew, color=bluec, lw=2)
    ax1.plot(x, nkc_bimod, label=label_bimod, color=pinkc, lw=2)

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 5.6)
    ax1.set_xlabel(xlabel=r"$X$ Value")
    ax1.set_ylabel("Density")

    ax1.set_title(r"$\,N\!-\!K\{C\}\,$ PDF")
    ax1.legend(prop=FontProperties(family="monospace", size=10))
    ax1.grid()
    mo.as_html(fig1).center()
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
def _(
    N_KC,
    bimod_params,
    bluec,
    grayc,
    greenc,
    lskew_params,
    mo,
    norm1_params,
    norm2_params,
    pinkc,
    plt,
    redc,
    rskew_params,
    x,
):
    fig2, ax2 = plt.subplots()
    NKC_norm1 = N_KC(x, *norm1_params)
    NKC_norm2 = N_KC(x, *norm2_params)
    NKC_rskew = N_KC(x, *rskew_params)
    NKC_lskew = N_KC(x, *lskew_params)
    NKC_bimod = N_KC(x, *bimod_params)

    ax2.plot(x, NKC_norm1, label="norm1", color=redc, lw=2)
    ax2.plot(x, NKC_norm2, label="norm2", color=grayc, lw=2)
    ax2.plot(x, NKC_rskew, label="rskew", color=greenc, lw=2)
    ax2.plot(x, NKC_lskew, label="lskew", color=bluec, lw=2)
    ax2.plot(x, NKC_bimod, label="bimod", color=pinkc, lw=2)

    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel(r"$X$ Value")
    ax2.set_ylabel("Probability")

    ax2.legend()
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.grid()
    mo.as_html(fig2).center()
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Fitting""")
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "./data/Global Air Quality (2024) - 6 Cities/New_York_Air_Quality.csv"
    )
    return (df,)


@app.cell
def _(calc_params_NKC, calc_params_beta, calc_params_ncbl, df, epsilon):
    raw_NO2 = df["NO2"].values
    adjusted_NO2 = raw_NO2 + epsilon
    scaled_NO2 = adjusted_NO2 / adjusted_NO2.max()
    scaled_NO2[scaled_NO2 >= 1.0] = 1.0 - epsilon

    nkc_ny_no2_params, nkc_ny_no2_ll = calc_params_NKC(scaled_NO2)
    beta_ny_no2_params, beta_ny_no2_ll = calc_params_beta(scaled_NO2)
    ncbl_ny_no2_params, ncbl_ny_no2_ll = calc_params_ncbl(scaled_NO2)
    return (
        beta_ny_no2_ll,
        beta_ny_no2_params,
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
    format_label,
    grayc,
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

    label_nkc = format_label(
        dict(zip(nkc_param_names, nkc_ny_no2_params)), label=r"$\,N\!-\!K\{C\}\,$ "
    )
    label_beta = format_label(
        dict(zip(beta_param_names, beta_ny_no2_params)), label=r"Beta $\ $   "
    )
    label_ncbl = format_label(
        dict(zip(ncbl_param_names, ncbl_ny_no2_params)),
        label=r"$\,N\!-\!CB\{L\}\,$",
    )

    ax3.plot(x, n_kc(x, *nkc_ny_no2_params), label=label_nkc, color=redc, lw=2)
    ax3.plot(
        x,
        beta.pdf(x, *beta_ny_no2_params),
        label=label_beta,
        color=bluec,
        lw=2,
        ls="--",
    )
    ax3.plot(
        x,
        n_cbl(x, *ncbl_ny_no2_params),
        label=label_ncbl,
        color=pinkc,
        lw=2,
        ls="--",
    )

    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 3.6)
    ax3.set_xlabel(r"$X$ Value")
    ax3.set_ylabel("Density")

    ax3.set_title("Nitrogen Dioxide Concentration in µg/m³")
    ax3.legend(prop=FontProperties(family="monospace", size=12))
    ax3.grid()
    mo.as_html(fig3).center()

    result_data = {
        "Distributions": ["NKC", "NCBL", "Beta"],
        "AIC": [
            AIC(5, nkc_ny_no2_ll, precision=2),
            AIC(4, ncbl_ny_no2_ll, precision=2),
            AIC(2, beta_ny_no2_ll, precision=2),
        ],
        "BIC": [
            BIC(5, raw_NO2.size, nkc_ny_no2_ll, precision=2),
            BIC(4, raw_NO2.size, ncbl_ny_no2_ll, precision=2),
            BIC(2, raw_NO2.size, beta_ny_no2_ll, precision=2),
        ],
    }
    df_res = pd.DataFrame(result_data)

    mo.hstack([mo.as_html(fig3), mo.as_html(df_res).style(width="800px")], justify='center', gap=5)
    return


if __name__ == "__main__":
    app.run()
