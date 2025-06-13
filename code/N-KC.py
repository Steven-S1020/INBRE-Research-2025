import marimo

__generated_with = "0.13.15"
app = marimo.App(width="full")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    from scipy.constants import pi
    from scipy.integrate import trapezoid
    from scipy.stats import norm, cauchy, skewnorm, beta, gamma
    from scipy.optimize import differential_evolution

    sns.set_theme(style='white', context='talk', font_scale=0.9, rc={
        'axes.edgecolor': 'black',
        'grid.color': 'silver'
    })
    plt.rcParams['figure.figsize'] = (3, 3)
    plt.rcParams['figure.dpi'] = 85
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    redc = '#a4031f'
    grayc = '#dddde3'
    greenc = '#88bf9b'
    bluec = '#3c91e6'
    pinkc = '#ffa0ac'
    return (
        beta,
        bluec,
        cauchy,
        differential_evolution,
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


@app.function
def F_R(x, a, b):
    return 1 - (1 - x**a)**b


@app.cell
def _(np):
    def f_R(x, a, b):
        safe_base = np.maximum(1 - x**a, 1e-12)
        result = safe_base ** (b - 1)
        return a * b * x ** (a - 1) * result
    return (f_R,)


@app.cell
def _(mo):
    mo.md(r"""## Y: Cauchy""")
    return


@app.cell
def _(cauchy):
    def Q_Y(x, gamma): # Quantile of Cauchy
        return cauchy.ppf(x, scale=gamma)
    return (Q_Y,)


@app.cell
def _(np, pi):
    def uQ_Y(x, gamma): # Derivative Quantile of Cauchy
        return gamma * (1/(np.cos(pi * (x - 0.5))**2)) * pi
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
        f_r = (2 * np.arctanh(1 - 2 * lamb) * (lamb ** x) * ((1 - lamb) ** (1 - x))) / (1 - 2 * lamb)
        F_r = ((lamb ** x) * ((1 - lamb) ** (1 - x)) + lamb - 1) / ((2 * lamb) - 1)
        s_r = 1 - F_r

        F_r = np.clip(F_r, epsilon, 1 - epsilon)
        s_r = np.clip(s_r, epsilon, 1 - epsilon)

        z = np.exp(-((alpha * np.log(F_r / (1 - F_r)) - mu) ** 2) / (2 * sig ** 2))

        numerator = z * alpha * (s_r + F_r) * f_r
        denominator = np.sqrt(2 * np.pi) * s_r * F_r * sig

        return numerator / denominator

    return (n_cbl,)


@app.cell
def _(Q_Y, f_R, f_T, uQ_Y):
    def n_kc(x, a, b, gamma, mu, sigma):
        return f_T(Q_Y(F_R(x, a, b), gamma), mu, sigma) * uQ_Y(F_R(x, a, b), gamma) * f_R(x, a, b)
    return (n_kc,)


@app.cell
def _(F_T, Q_Y):
    def N_KC(x, a, b, gamma, mu, sigma):
        return F_T(Q_Y(F_R(x, a, b), gamma), mu, sigma)
    return (N_KC,)


@app.cell
def _(mo):
    mo.md(r"""## Calculations""")
    return


@app.cell
def _(differential_evolution, n_kc, np):
    epsilon = 1e-16

    def calc_params_NKC(data):
        def ll_nkc(params):
            a, b, gamma, mu, sigma = params
            nkc_vals = n_kc(data, a, b, gamma, mu, sigma) + epsilon
            return -np.sum(np.log(nkc_vals))

        bounds = [
            (1e-12, 2.5),    # a
            (1e-12, 2.5),    # b
            (1e-12, 30),     # lam
            (-30, 30),       # mu
            (1e-12, 30)      # sigma
        ]

        # Run global optimization
        result = differential_evolution(
            ll_nkc,
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
        return result.x, -result.fun
    return calc_params_NKC, epsilon


@app.cell
def _(differential_evolution, epsilon, gamma, np):
    def calc_params_gamma(data):
        def ll_gamma(params):
            shape, scale = params
            pdf_vals = gamma.pdf(data, a=shape, scale=scale) + epsilon
            return -np.sum(np.log(pdf_vals))

        bounds = [
            (1e-5, 100),   # shape
            (1e-5, 100),   # scale
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
            pdf_vals = beta.pdf(data, a, b) + epsilon
            return -np.sum(np.log(pdf_vals))

        bounds = [
            (1e-5, 100),  # a
            (1e-5, 100),  # b
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
            tol=1e-4,
            updating="immediate",
            )
        return result.x, -result.fun
    return (calc_params_beta,)


@app.cell
def _(differential_evolution, epsilon, n_cbl, np):
    def calc_params_ncbl(data):
        def ll_ncbl(params):
            mu, sig, alpha, lamb = params
            pdf_vals = n_cbl(data, mu, sig, alpha, lamb) + epsilon
            return -np.sum(np.log(pdf_vals))

        bounds = [
            (-100, 100),     # mu
            (1e-5, 100),     # sigma
            (1e-5, 100),     # alpha
            (1e-5, 1 - 1e-5) # lambda
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


@app.cell
def _(mo):
    mo.md(r"""## PDF & CDF of $\,N\!-\!K\{C\}\,$""")
    return


@app.cell
def _(calc_params_NKC, norm, np, skewnorm):
    x = np.linspace(1e-12,1 - 1e-12, 1000000)

    norm1 = np.clip(norm.rvs(loc=0.5, scale=0.09, size=1000), 1e-16, 1 - 1e-16)
    norm2 = np.clip(norm.rvs(loc=0.5, scale=0.15, size=1000), 1e-16, 1 - 1e-16)
    rskew = np.clip(skewnorm.rvs(a=25, loc=0.1, scale=0.25, size=1000), 1e-16, 1 - 1e-16)
    lskew = np.clip(skewnorm.rvs(a=-25, loc=0.9, scale=0.25, size=1000), 1e-16, 1 - 1e-16)
    bimod = np.clip(np.concatenate([
        norm.rvs(loc=0.3, scale=0.065, size=500),
        norm.rvs(loc=0.7, scale=0.065, size=500)
    ]), 1e-16, 1 - 1e-16)

    norm1_params, norm1_ll = calc_params_NKC(norm1)
    print(f"norm1:\n  Params: {norm1_params}\n  Log-Likelihood: {norm1_ll:.4f}")

    norm2_params, norm2_ll = calc_params_NKC(norm2)
    print(f"norm2:\n  Params: {norm2_params}\n  Log-Likelihood: {norm2_ll:.4f}")

    rskew_params, rskew_ll = calc_params_NKC(rskew)
    print(f"rskew:\n  Params: {rskew_params}\n  Log-Likelihood: {rskew_ll:.4f}")

    lskew_params, lskew_ll = calc_params_NKC(lskew)
    print(f"lskew:\n  Params: {lskew_params}\n  Log-Likelihood: {lskew_ll:.4f}")

    bimod_params, bimod_ll = calc_params_NKC(bimod)
    print(f"bimod:\n  Params: {bimod_params}\n  Log-Likelihood: {bimod_ll:.4f}")
    return (
        bimod_params,
        lskew_params,
        norm1_params,
        norm2_params,
        rskew_params,
        x,
    )


@app.cell
def _(mo):
    mo_a = mo.ui.slider(start=0.001, stop=2, step=0.001, value=1.27, label=fr'$a$')
    mo_b = mo.ui.slider(start=0.001, stop=10, step=0.001, value=1.22, label=fr'$b$')
    mo_gamma = mo.ui.slider(start=0.1, stop=100, step=0.1, value=58.6, label=fr'$\gamma$')
    mo_mu = mo.ui.slider(start=-100, stop=100, step=0.001, value=-2, label=fr'$\mu$')
    mo_sigma = mo.ui.slider(start=0.001, stop=100, step=0.001, value=66.9, label=fr'$\sigma$')

    mo.hstack([mo_a, mo_b, mo_gamma, mo_mu, mo_sigma], justify='center', align='center', gap=5)
    return


@app.cell
def _(
    bimod_params,
    bluec,
    grayc,
    greenc,
    lskew_params,
    mo,
    n_kc,
    norm1_params,
    norm2_params,
    pinkc,
    plt,
    redc,
    rskew_params,
    x,
):

    fig1, ax1 = plt.subplots()

    nkc_norm1 = n_kc(x, *norm1_params)
    nkc_norm2 = n_kc(x, *norm2_params)
    nkc_rskew = n_kc(x, *rskew_params)
    nkc_lskew = n_kc(x, *lskew_params)
    nkc_bimod = n_kc(x, *bimod_params)

    ax1.plot(x, nkc_norm1, label='norm1', color=redc, lw=2)
    ax1.plot(x, nkc_norm2, label='norm2', color=grayc, lw=2)
    ax1.plot(x, nkc_rskew,  label='rskew',  color=greenc, lw=2)
    ax1.plot(x, nkc_lskew, label='lskew',  color=bluec, lw=2)
    ax1.plot(x, nkc_bimod, label='bimod',  color=pinkc, lw=2)

    #sns.histplot(bimod, bins=70, stat='density', ax=ax1, color=grayc)
    #ax1.plot(x, n_kc(x, mo_a.value, mo_b.value, mo_gamma.value, mo_mu.value, mo_sigma.value), lw=3)

    title = r'$\,N\!-\!K\{C\}\,$ PDF'
    ax1.set_title(title)
    ax1.set_xlabel(xlabel=r'$X$ Value')
    ax1.set_ylabel('Probability')

    ax1.set_xlim(0,1)
    ax1.set_ylim(0,5)
    ax1.grid()
    mo.as_html(fig1).style(
        display="block",
        margin="auto",
        height='800px',
        width="800px",
        box_shadow="0 0 4px rgba(0,0,0,0.2)",
        border_radius="8px"
    )
    return nkc_bimod, nkc_lskew, nkc_norm1, nkc_norm2, nkc_rskew, title


@app.cell
def _(
    N_KC,
    bimod_params,
    bluec,
    grayc,
    greenc,
    lskew_params,
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

    ax2.plot(x, NKC_norm1, label='norm1 CDF', color=redc, lw=2)
    ax2.plot(x, NKC_norm2, label='norm2 CDF', color=grayc, lw=2)
    ax2.plot(x, NKC_rskew, label='rskew CDF', color=greenc, lw=2)
    ax2.plot(x, NKC_lskew, label='lskew CDF', color=bluec, lw=2)
    ax2.plot(x, NKC_bimod, label='bimod CDF', color=pinkc, lw=2)

    title2 = r'$\,N\!-\!K\{C\}\,$ CDF'
    ax2.set_title(title2)
    ax2.set_xlabel(xlabel=r'$X$ Value')
    ax2.set_ylabel('Probability')

    ax2.legend()
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.grid()
    fig2.gca()
    return


@app.cell
def _(nkc_bimod, nkc_lskew, nkc_norm1, nkc_norm2, nkc_rskew, trapezoid, x):
    nkc_norm1[nkc_norm1 <= 0] = 1e-8
    print('Area under nkc_norm1:', trapezoid(nkc_norm1, x))

    nkc_norm2[nkc_norm2 <= 0] = 1e-8
    print('Area under nkc_norm2:', trapezoid(nkc_norm2, x))

    nkc_lskew[nkc_lskew <= 0] = 1e-8
    print('Area under nkc_lskew:', trapezoid(nkc_lskew, x))

    nkc_rskew[nkc_rskew <= 0] = 1e-8
    print('Area under nkc_rskew:', trapezoid(nkc_rskew, x))

    nkc_bimod[nkc_bimod <= 0] = 1e-8
    print('Area under nkc_bimod:', trapezoid(nkc_bimod, x))
    return


@app.cell
def _(mo):
    mo.md(r"""## Data Fitting""")
    return


@app.cell
def _(pd):
    df = pd.read_csv('./data/Global Air Quality (2024) - 6 Cities/New_York_Air_Quality.csv')
    df.head()
    return (df,)


@app.cell
def _(calc_params_NKC, calc_params_beta, calc_params_ncbl, df, epsilon):
    raw_NO2 = df['NO2'].values
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
        scaled_NO2,
    )


@app.cell
def _(
    beta,
    beta_ny_no2_ll,
    beta_ny_no2_params,
    bluec,
    grayc,
    n_cbl,
    n_kc,
    ncbl_ny_no2_ll,
    ncbl_ny_no2_params,
    nkc_ny_no2_ll,
    nkc_ny_no2_params,
    pinkc,
    plt,
    redc,
    scaled_NO2,
    sns,
    title,
    x,
):
    fig3, ax3 = plt.subplots()
    sns.histplot(scaled_NO2, bins=70, stat='density', ax=ax3, color=grayc)

    label_nkc = fr'$\,N\!-\!K\{{C\}}\,$  logL={nkc_ny_no2_ll:.2f},  ' \
            fr'$\alpha={nkc_ny_no2_params[0]:.2f}$, ' \
            fr'$\beta={nkc_ny_no2_params[1]:.2f}$, ' \
            fr'$\lambda={nkc_ny_no2_params[2]:.2f}$, ' \
            fr'$\mu={nkc_ny_no2_params[3]:.2f}$, ' \
            fr'$\sigma={nkc_ny_no2_params[4]:.2f}$'
    label_beta = fr'Beta  logL={beta_ny_no2_ll:.2f},  ' \
                 fr'$\alpha={beta_ny_no2_params[0]:.2f}$, ' \
                 fr'$\beta={beta_ny_no2_params[1]:.2f}$'
    label_ncbl = fr'N-CBL  logL={ncbl_ny_no2_ll:.2f},  ' \
                 fr'$\mu={ncbl_ny_no2_params[0]:.2f}$, ' \
                 fr'$\sigma={ncbl_ny_no2_params[1]:.2f}$, ' \
                 fr'$\alpha={ncbl_ny_no2_params[2]:.2f}$, ' \
                 fr'$\lambda={ncbl_ny_no2_params[3]:.2f}$'


    ax3.plot(x, n_kc(x, *nkc_ny_no2_params), label=label_nkc, color=redc, lw=2)
    ax3.plot(x, beta.pdf(x, *beta_ny_no2_params), label=label_beta, color=bluec, lw=2, ls='--')
    ax3.plot(x, n_cbl(x, *ncbl_ny_no2_params), label=label_ncbl, color=pinkc, lw=2, ls='--')

    title3 = r'$\,N\!-\!K\{C\}\,$ fit to Air Polution Data'
    ax3.set_title(title, fontsize=28, pad=20)
    ax3.set_xlabel(xlabel=r'$X$ Value', fontsize=28, labelpad=20)
    ax3.set_ylabel('Density', fontsize=28, labelpad=20)
    ax3.set_xlim(0,1)
    ax3.grid()
    ax3.legend(fontsize=16)
    ax3.tick_params(labelsize=20)
    fig3.gca()
    return


if __name__ == "__main__":
    app.run()
