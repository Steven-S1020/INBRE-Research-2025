import marimo

__generated_with = "0.13.6"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from math import sqrt
    from scipy.special import erf, gammainc
    from scipy.special import gamma as gamma_fun
    from scipy.optimize import minimize, differential_evolution
    from scipy.constants import pi
    from scipy.stats import gamma, norm, cauchy, expon, beta, skewnorm, weibull_min
    from scipy.integrate import quad, trapezoid

    sns.set_theme(style='white', context='talk', font_scale=1.25, rc={
        'axes.edgecolor': 'black',
        'grid.color': 'silver'
    })

    plt.rcParams['figure.figsize'] = (15,13)
    plt.rcParams['figure.dpi'] = 85
    return (
        cauchy,
        differential_evolution,
        mo,
        norm,
        np,
        pi,
        plt,
        skewnorm,
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
    def calc_params_NKC(data):
        # Log-likelihood function with zero protection
        def ll(params, data):
            a, b, gamma, mu, sigma = params
            nkc_vals = n_kc(data, a, b, gamma, mu, sigma)
            nkc_vals[nkc_vals <= 0] = 1e-12
            return -np.sum(np.log(nkc_vals))

        # Bounds (tune if needed)
        bounds = [
            (1e-12, 2.5),    # a
            (1e-12, 2.5),    # b
            (1e-12, 100),    # lam
            (-100, 100),     # mu
            (1e-12, 100)     # sigma
        ]

        # Run global optimization
        result = differential_evolution(
            lambda p: ll(p, data),
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

        #print("Best log-likelihood:", -result.fun)
        print(result.x)
        return result.x
    return (calc_params_NKC,)


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

    norm1_params = calc_params_NKC(norm1)
    norm2_params = calc_params_NKC(norm2)
    rskew_params = calc_params_NKC(rskew)
    lskew_params = calc_params_NKC(lskew)
    bimod_params = calc_params_NKC(bimod)
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
    lskew_params,
    n_kc,
    norm1_params,
    norm2_params,
    plt,
    rskew_params,
    x,
):
    redc = '#a4031f'
    grayc = '#dddde3'
    greenc = '#88bf9b'
    bluec = '#3c91e6'
    pinkc = '#ffa0ac'

    fig, ax = plt.subplots()

    nkc_norm1 = n_kc(x, *norm1_params)
    nkc_norm2 = n_kc(x, *norm2_params)
    nkc_rskew = n_kc(x, *rskew_params)
    nkc_lskew = n_kc(x, *lskew_params)
    nkc_bimod = n_kc(x, *bimod_params)

    ax.plot(x, nkc_norm1, label='norm1', color=redc, lw=2)
    ax.plot(x, nkc_norm2, label='norm2', color=grayc, lw=2)
    ax.plot(x, nkc_rskew,  label='rskew',  color=greenc, lw=2)
    ax.plot(x, nkc_lskew, label='lskew',  color=bluec, lw=2)
    ax.plot(x, nkc_bimod, label='bimod',  color=pinkc, lw=2)

    #sns.histplot(bimod, bins=70, stat='density', ax=ax, color=grayc)
    #ax.plot(x, n_kc(x, mo_a.value, mo_b.value, mo_gamma.value, mo_mu.value, mo_sigma.value), lw=3)

    title = r'$\,N\!-\!K\{C\}\,$ PDF'
    ax.set_title(title, fontsize=28, pad=20)
    ax.set_xlabel(xlabel=r'$X$ Value', fontsize=28, labelpad=20)
    ax.set_ylabel('Probability', fontsize=28, labelpad=20)

    ax.legend(fontsize=16)
    ax.set_xlim(0,1)
    ax.set_ylim(0,5)
    ax.grid()
    ax.tick_params(labelsize=20)
    fig.gca()
    return (
        bluec,
        grayc,
        greenc,
        nkc_bimod,
        nkc_lskew,
        nkc_norm1,
        nkc_norm2,
        nkc_rskew,
        pinkc,
        redc,
    )


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
    ax2.set_title(title2, fontsize=28, pad=20)
    ax2.set_xlabel(xlabel=r'$X$ Value', fontsize=28, labelpad=20)
    ax2.set_ylabel('Probability', fontsize=28, labelpad=20)

    ax2.legend(fontsize=16)
    ax2.set_xlim(0,1)
    ax2.set_ylim(0,1)
    ax2.grid()
    ax2.tick_params(labelsize=20)
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


if __name__ == "__main__":
    app.run()
