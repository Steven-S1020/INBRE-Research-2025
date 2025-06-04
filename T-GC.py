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
    from scipy.stats import gamma, norm, cauchy, expon, beta, skewnorm
    from scipy.integrate import quad, trapezoid

    sns.set_theme(style='white', context='poster', font_scale=1.25, palette='pastel')

    plt.rcParams['figure.figsize'] = (15,13)
    plt.rcParams['figure.dpi'] = 85
    return cauchy, gamma, mo, norm, np, pi, plt, trapezoid


@app.cell
def _(mo):
    mo.md(r"""## R: Gamma""")
    return


@app.cell
def _(gamma):
    def F_R(x, alpha, theta): # CDF of Gamma
        return gamma.cdf(x, a=alpha, scale=theta)
    return (F_R,)


@app.cell
def _(gamma):
    def f_R(x, alpha, theta): # PDF of Gamma
        return gamma.pdf(x, a=alpha, scale=theta)
    return (f_R,)


@app.cell
def _(mo):
    mo.md(r"""## Y: Cauchy""")
    return


@app.cell
def _(cauchy):
    def Q_Y(x, lam): # Quantile of Cauchy
        return cauchy.ppf(x, scale=lam)
    return (Q_Y,)


@app.cell
def _(np, pi):
    def uQ_Y(x, lam): # Derivative Quantile of Cauchy
        return lam * (1/(np.cos(pi * (x - 0.5))**2)) * pi
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
def _(F_R, Q_Y, f_R, f_T, uQ_Y):
    def n_gc(x, alpha, theta, mu, sigma, lam): # Normal - Gamma {Cauchy}
        return f_T(Q_Y(F_R(x, alpha, theta), lam), mu, sigma) * uQ_Y(F_R(x, alpha, theta), lam) * f_R(x, alpha, theta)
    return (n_gc,)


@app.cell
def _(F_R, F_T, Q_Y):
    def N_GC(x, alpha, theta, mu, sigma, lam): # Normal - Gamma {Cauchy}
        return F_T(Q_Y(F_R(x, alpha, theta), lam), mu, sigma)
    return


@app.cell
def _(mo):
    mo.md(r"""## Calculations""")
    return


app._unparsable_cell(
    r"""
    # Your data
    data = np.random.normal(10, 2, 5000)
    #norm2 = np.random.normal(15, 1, 500)
    #data = skewnorm.rvs(-1000, loc=19.5, scale=4, size=2000)
    #data = np.concatenate((norm1, norm2))

    # Log-likelihood function with zero protection
    def ll(params, data):
        alpha, theta, mu, sigma, lam = params
        ngc_vals = n_gc(data, alpha, theta, mu, sigma, lam)
        ngc_vals[ngc_vals <= 0] = 1e-12
        return -np.sum(np.log(ngc_vals))

    # Bounds (tune if needed)
    bounds = [
        (1e-5, 10),     # alpha
        (1e-5, 10),     # theta
        (0, 20),        # mu
        (0.1, 20),      # sigma
        (0.01, 100)     # lam
    ]

    # Run global optimization
    #result = differential_evolution(
        lambda p: ll(p, data),
        bounds,
        strategy=\"best1bin\",     # Good default mutation strategy
        maxiter=1000,            # 🚀 Increase number of generations (default is 100)
        popsize=20,              # 🚀 Increase number of individuals per generation (default is 15)
        mutation=(0.5, 1.0),     # Mutation strength (can widen range to explore more)
        recombination=0.7,       # Crossover probability (how much mixing between solutions)
        polish=True,             # Fine-tune best solution with L-BFGS-B at the end
        tol=1e-7,                # Convergence tolerance (lower = more precise)
        updating=\"immediate\",   # Update as soon as better solution is found (more aggressive)
        )

    print(\"Best log-likelihood:\", -result.fun)
    print(\"Best parameters:\", result.x)
    """,
    name="_"
)


@app.cell
def _(mo):
    mo_alpha = mo.ui.slider(start=0, stop=5, step=0.01, value=2, label=fr'$\alpha$')
    mo_theta = mo.ui.slider(start=0, stop=30, step=0.0001, value=3, label=fr'$\theta$')
    mo_mu    = mo.ui.slider(start=-100, stop=200, step=0.01, value=16.28, label=fr'$\mu$')
    mo_sigma    = mo.ui.slider(start=0, stop=1000, step=0.01, value=10.39, label=fr'$\sigma$')
    mo_lam = mo.ui.slider(start=0, stop=50, step=0.001, value=0.001, label=fr'$\lambda$')

    mo.hstack([mo_alpha, mo_theta, mo_mu, mo_sigma, mo_lam], justify='center', align='center', gap=5)
    return mo_alpha, mo_lam, mo_mu, mo_sigma, mo_theta


@app.cell
def _(mo_alpha, mo_lam, mo_mu, mo_sigma, mo_theta, n_gc, norm, np, plt):
    s_alpha = mo_alpha.value
    s_theta = mo_theta.value
    s_mu    = mo_mu.value
    s_sigma = mo_sigma.value
    s_lam = mo_lam.value

    #alpha = result.x[0]
    #theta = result.x[1]
    #mu    = result.x[2]
    #sigma = result.x[3]
    #lam   = result.x[4]

    x = np.linspace(0,1000,10000)
    y_norm = norm.pdf(x, loc=5, scale=1)

    fig, ax = plt.subplots()

    #sns.histplot(data, bins=16, stat='density', ax=ax)

    #ax.plot(x, n_gc(x, s_alpha, s_theta, s_mu, s_sigma, s_lam), c='red', label='Slider')
    #ax.plot(x, n_gc(x, alpha, theta, mu, sigma, lam), c='blue', label='MLE')

    ngc_norm1 = n_gc(x, 0.93, 10, 15.25, 3.95, 27.4)
    ngc_norm2 = n_gc(x, 0.91, 10, 19.9, 10.5, 34)
    ngc_lskew = n_gc(x, 6.5, 1.9, 17, 10, 9)
    ngc_rskew = n_gc(x, 0.6, 10, 0, 18, 26)
    ngc_bimodal = n_gc(x, 4.6, 2, 8, 20, 4.67)

    ax.plot(x, ngc_norm1, c='red', lw=2)
    ax.plot(x, ngc_norm2, c='blue', lw=2)
    ax.plot(x, ngc_lskew, c='green', lw=2)
    ax.plot(x, ngc_rskew, c='purple', lw=2)
    ax.plot(x, ngc_bimodal, c='orange', lw=2)


    #ax.legend()
    ax.set_xlim(0,20)
    ax.set_ylim(0,0.45)
    ax.grid()
    ax.tick_params(labelsize=20)
    fig.gca()
    return ngc_bimodal, ngc_lskew, ngc_norm1, ngc_norm2, ngc_rskew, x


@app.cell
def _(ngc_bimodal, ngc_lskew, ngc_norm1, ngc_norm2, ngc_rskew, trapezoid, x):
    ngc_norm1[ngc_norm1 <= 0] = 1e-8
    print('Area under ngc_norm1:', trapezoid(ngc_norm1, x))

    ngc_norm2[ngc_norm2 <= 0] = 1e-8
    print('Area under ngc_norm2:', trapezoid(ngc_norm2, x))

    ngc_lskew[ngc_lskew <= 0] = 1e-8
    print('Area under ngc_lskew:', trapezoid(ngc_lskew, x))

    ngc_rskew[ngc_rskew <= 0] = 1e-8
    print('Area under ngc_rskew:', trapezoid(ngc_rskew, x))

    ngc_bimodal[ngc_bimodal <= 0] = 1e-8
    print('Area under ngc_bimodal:', trapezoid(ngc_bimodal, x))

    trapezoid(ngc_rskew, x)
    return


if __name__ == "__main__":
    app.run()
