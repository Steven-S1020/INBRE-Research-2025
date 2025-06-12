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

    from scipy.constants import pi
    from scipy.integrate import trapezoid
    from scipy.stats import norm, cauchy, skewnorm, beta, gamma
    from scipy.optimize import differential_evolution
    from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
    from matplotlib.font_manager import FontProperties

    redc = '#a4031f'
    grayc = '#dddde3'
    greenc = '#88bf9b'
    bluec = '#3c91e6'
    pinkc = '#ffa0ac'

    sns.set_theme(style='white', context='talk', font_scale=1.25, rc={
        'axes.edgecolor': 'black',
        'grid.color': 'silver'
    })
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['lines.linewidth'] = 1.5
    plt.rcParams['axes.labelsize'] = 20
    plt.rcParams['axes.titlesize'] = 26
    plt.rcParams['axes.titlepad'] = 20
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['xtick.minor.size'] = 6
    plt.rcParams['ytick.minor.size'] = 6
    plt.rcParams['xtick.color'] = 'black'
    plt.rcParams['ytick.color'] = 'black'

    epsilon = 1e-8
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


@app.function
def F_R(x, a, b):
    return 1 - (1 - x**a)**b


@app.function
def f_R(x, a, b):
    return a * b * (x ** (a - 1)) * (1 - x**a)**(b - 1)


@app.cell
def _(mo):
    mo.md(r"""## Y: Cauchy""")
    return


@app.cell
def _(cauchy, epsilon, np):
    def Q_Y(x, gamma, min, max): # Quantile of Cauchy
        p = np.clip(x, epsilon, 1 - epsilon)
        q = cauchy.ppf(p, scale=gamma)
        return np.clip(q, min, max)
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
def _(Q_Y, f_T, uQ_Y):
    def n_kc(x, a, b, gamma, mu, sigma):
        return f_T(Q_Y(F_R(x, a, b), gamma, (mu - 5 * sigma), (mu + 5 * sigma)), mu, sigma) * uQ_Y(F_R(x, a, b), gamma) * f_R(x, a, b)
    return (n_kc,)


@app.cell
def _(F_T, Q_Y):
    def N_KC(x, a, b, gamma, mu, sigma):
        return F_T(Q_Y(F_R(x, a, b), gamma, (mu - 5 * sigma), (mu + 5 * sigma)), mu, sigma)
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
            nkc_vals = n_kc(data, a, b, gamma, mu, sigma)
            if np.any(nkc_vals == 0):
                print('Params when nkc_vals == 0', params)
            nkc_vals[nkc_vals <= 0] = epsilon
            return -np.sum(np.log(nkc_vals))

        bounds = [
            (epsilon, 2.5),    # a
            (epsilon, 2.5),    # b
            (epsilon, 30),     # lam
            (-30, 30),         # mu
            (epsilon, 30)      # sigma
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
    return (calc_params_NKC,)


@app.cell
def _(differential_evolution, gamma, np):
    def calc_params_gamma(data):
        def ll_gamma(params):
            shape, scale = params
            pdf_vals = gamma.pdf(data, a=shape, scale=scale)
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
def _(beta, differential_evolution, np):
    def calc_params_beta(data):
        def ll_beta(params):
            a, b = params
            pdf_vals = beta.pdf(data, a, b)
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
def _(differential_evolution, n_cbl, np):
    def calc_params_ncbl(data):
        def ll_ncbl(params):
            mu, sig, alpha, lamb = params
            pdf_vals = n_cbl(data, mu, sig, alpha, lamb)
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
    mo.md(r"""## Helper Functions""")
    return


@app.cell
def _():
    nkc_param_names = ['a', 'b', '\\lambda', '\\mu', '\\sigma']
    ncbl_param_names = ['\\mu', '\\sigma', '\\alpha', '\\lambda']
    beta_param_names = ['\\alpha', '\\beta']


    def format_label(params, label="", precision=2, param_width=10):
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
            raise TypeError('params must be a dict')

        param_strs = []
        for k, v in params.items():
            raw = f"${k}=${v:.{precision}f}"
            n = (param_width - len(raw)) if '\\' not in k else (param_width - (len(raw) - len(k)))
            padded = raw + (' ' * n)
            param_strs.append(padded)

        return f"{label}  " + "".join(param_strs)
    return beta_param_names, format_label, ncbl_param_names, nkc_param_names


@app.cell
def _(mo):
    mo.md(r"""## PDF & CDF of $\,N\!-\!K\{C\}\,$""")
    return


@app.cell
def _(calc_params_NKC, epsilon, norm, np, skewnorm):
    x = np.linspace(epsilon,1 - epsilon, 10000)

    norm1 = norm.rvs(loc=0, scale=1, size=1000)
    norm2 = norm.rvs(loc=0, scale=1.5, size=1000)
    rskew = skewnorm.rvs(a=25, loc=0, scale=1, size=1000)
    lskew = skewnorm.rvs(a=-25, loc=0, scale=1, size=1000)
    bimod = np.concatenate([ norm.rvs(loc=0, scale=0.5, size=500), norm.rvs(loc=1, scale=0.5, size=500)])

    adjusted_norm1 = norm1 + abs(norm1.min()) + epsilon
    adjusted_norm2 = norm2 + abs(norm2.min()) + epsilon
    adjusted_rskew = rskew + abs(rskew.min()) + epsilon
    adjusted_lskew = lskew + abs(lskew.min()) + epsilon
    adjusted_bimod = bimod + abs(bimod.min()) + epsilon

    scaled_norm1 = adjusted_norm1 / (adjusted_norm1.max() + epsilon)
    scaled_norm2 = adjusted_norm2 / (adjusted_norm2.max() + epsilon)
    scaled_rskew = adjusted_rskew / (adjusted_rskew.max() + epsilon)
    scaled_lskew = adjusted_lskew / (adjusted_lskew.max() + epsilon)
    scaled_bimod = adjusted_bimod / (adjusted_bimod.max() + epsilon)

    norm1_params, norm1_ll = calc_params_NKC(scaled_norm1)
    norm2_params, norm2_ll = calc_params_NKC(scaled_norm2)
    rskew_params, rskew_ll = calc_params_NKC(scaled_rskew)
    lskew_params, lskew_ll = calc_params_NKC(scaled_lskew)
    bimod_params, bimod_ll = calc_params_NKC(scaled_bimod)
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

    mo.hstack([mo_a, mo_b, mo_gamma, mo_mu, mo_sigma], justify='center', align='center', gap=2.5)
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
    n_kc,
    nkc_param_names,
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

    label_norm1 = format_label(dict(zip(nkc_param_names, norm1_params)), label='norm1')
    label_norm2 = format_label(dict(zip(nkc_param_names, norm2_params)), label='norm2')
    label_rskew = format_label(dict(zip(nkc_param_names, rskew_params)), label='rskew')
    label_lskew = format_label(dict(zip(nkc_param_names, lskew_params)), label='lskew')
    label_bimod = format_label(dict(zip(nkc_param_names, bimod_params)), label='bimod')

    #sns.histplot(scaled_rskew, bins=70, stat='density', ax=ax1, color=grayc)

    ax1.plot(x, nkc_norm1, label=label_norm1, color=redc, lw=2)
    ax1.plot(x, nkc_norm2, label=label_norm2, color=grayc, lw=2)
    ax1.plot(x, nkc_rskew,  label=label_rskew,  color=greenc, lw=2)
    ax1.plot(x, nkc_lskew, label=label_lskew,  color=bluec, lw=2)
    ax1.plot(x, nkc_bimod, label=label_bimod,  color=pinkc, lw=2)

    ax1.set_xlim(0,1)
    ax1.set_ylim(0,5.6)
    ax1.set_xlabel(xlabel=r'$X$ Value')
    ax1.set_ylabel('Density')

    ax1.set_title(r'$\,N\!-\!K\{C\}\,$ PDF')
    ax1.legend(prop=FontProperties(family='monospace', size=12))
    ax1.grid()
    mo.as_html(fig1).center()
    return nkc_bimod, nkc_lskew, nkc_norm1, nkc_norm2, nkc_rskew


@app.cell
def _(
    FontProperties,
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

    ax2.plot(x, NKC_norm1, label='norm1 CDF', color=redc, lw=2)
    ax2.plot(x, NKC_norm2, label='norm2 CDF', color=grayc, lw=2)
    ax2.plot(x, NKC_rskew, label='rskew CDF', color=greenc, lw=2)
    ax2.plot(x, NKC_lskew, label='lskew CDF', color=bluec, lw=2)
    ax2.plot(x, NKC_bimod, label='bimod CDF', color=pinkc, lw=2)

    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.set_xlabel(r'$X$ Value')
    ax2.set_ylabel('Probability')

    ax2.set_title(r'$\,N\!-\!K\{C\}\,$ CDF')
    ax2.legend(prop=FontProperties(family='monospace', size=12))
    ax2.grid()
    mo.as_html(fig2).center()
    return


@app.cell
def _(nkc_bimod, nkc_lskew, nkc_norm1, nkc_norm2, nkc_rskew, trapezoid, x):
    print('Area under nkc_norm1:', trapezoid(nkc_norm1, x))
    print('Area under nkc_norm2:', trapezoid(nkc_norm2, x))
    print('Area under nkc_lskew:', trapezoid(nkc_lskew, x))
    print('Area under nkc_rskew:', trapezoid(nkc_rskew, x))
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
    adjusted_NO2 = raw_NO2 + abs(raw_NO2.min()) + epsilon
    adjusted_max = adjusted_NO2.max() + epsilon
    scaled_NO2 = adjusted_NO2 / adjusted_max

    nkc_ny_no2_params, nkc_ny_no2_ll = calc_params_NKC(scaled_NO2)
    beta_ny_no2_params, beta_ny_no2_ll = calc_params_beta(scaled_NO2)
    ncbl_ny_no2_params, ncbl_ny_no2_ll = calc_params_ncbl(scaled_NO2)
    return (
        beta_ny_no2_params,
        ncbl_ny_no2_params,
        nkc_ny_no2_params,
        raw_NO2,
        scaled_NO2,
    )


@app.cell
def _(Q_Y, f_T, n_kc, np, plt, x):
    a = 0.18612344
    b = 0.28285843
    gamma3 = 25.96348543
    mu = -14.74427067
    sigma = 9.78482528

    term1_1 = F_R(x, a, b)
    term1_2 = Q_Y(term1_1, gamma3, (mu - 5 * sigma), (mu + 5 * sigma))
    term1_3 = f_T(term1_2, mu, sigma)
    test_nkc = n_kc(x, 0.18612344, 0.28285843, 25.96348543, -14.74427067, 9.78482528)

    print('term1_2 where f_T == 0:', term1_2[term1_3 == 0])
    print('F_R(x) near edges:', term1_1[term1_3 == 0])
    print('x:', np.any(x == 0))
    print('term1:', np.any(term1_1 == 0))
    print('term2:', np.any(term1_2 == 0))
    print('term3:', np.any(term1_3 == 0))
    print('test_nkc', np.any(test_nkc == 0))

    #plt.plot(x, n_kc(x, 1.65, 3.48, 6.8, 1.76, 10))
    plt.plot(x, test_nkc)
    return gamma3, mu, sigma


@app.cell
def _(Q_Y, gamma3, mu, sigma):
    Q_Y(.8, gamma3, (mu - 5 * sigma), (mu + 5 * sigma))
    return


@app.cell
def _(np, raw_NO2, scaled_NO2):
    idxs = np.where(scaled_NO2 == 1.0)
    print("Indexes where scaled == 1.0:", idxs)
    print("Original data at those indexes:", raw_NO2[idxs])
    print("Max scaled value:", scaled_NO2.max())
    print("Is it very close to 1?", np.isclose(scaled_NO2.max(), 1.0, rtol=0, atol=1e-12))
    return


@app.cell
def _(np, scaled_NO2):
    np.set_printoptions(threshold=np.inf)
    print(np.sort(scaled_NO2))
    return


@app.cell
def _(
    FontProperties,
    beta,
    beta_ny_no2_params,
    beta_param_names,
    bluec,
    format_label,
    grayc,
    mo,
    n_cbl,
    n_kc,
    ncbl_ny_no2_params,
    ncbl_param_names,
    nkc_ny_no2_params,
    nkc_param_names,
    pinkc,
    plt,
    redc,
    scaled_NO2,
    sns,
    x,
):
    fig3, ax3 = plt.subplots()
    sns.histplot(scaled_NO2, bins=70, stat='density', ax=ax3, color=grayc)

    label_nkc  = format_label(dict(zip(nkc_param_names, nkc_ny_no2_params)), label=r'$\,N\!-\!K\{C\}\,$  ')
    label_beta = format_label(dict(zip(beta_param_names, beta_ny_no2_params)), label='Beta           ')
    label_ncbl = format_label(dict(zip(ncbl_param_names, ncbl_ny_no2_params)), label=r'$\,N\!-\!CB\{L\}\,$')

    ax3.plot(x, n_kc(x, *nkc_ny_no2_params), label=label_nkc, color=redc, lw=2)
    ax3.plot(x, beta.pdf(x, *beta_ny_no2_params), label=label_beta, color=bluec, lw=2, ls='--')
    ax3.plot(x, n_cbl(x, *ncbl_ny_no2_params), label=label_ncbl, color=pinkc, lw=2, ls='--')

    ax3.set_xlim(0,1)
    ax3.set_ylim(0,3.6)
    ax3.set_xlabel(r'$X$ Value')
    ax3.set_ylabel('Density')

    ax3.set_title(r'$\,N\!-\!K\{C\}\,$ fit to Air Polution Data')
    ax3.legend(prop=FontProperties(family='monospace', size=12))
    ax3.grid()
    mo.as_html(fig3).center()
    return


if __name__ == "__main__":
    app.run()
