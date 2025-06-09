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
        expon,
        gamma,
        mo,
        norm,
        np,
        pi,
        plt,
        skewnorm,
        sns,
        trapezoid,
        weibull_min,
    )


@app.cell
def _(mo):
    mo.md(r"""## R: Gamma""")
    return


@app.function
def F_R_ks(x, a, b):
    return 1 - (1 - x**a)**b


@app.cell
def _(np):
    def f_R_ks(x, a, b):
        safe_base = np.maximum(1 - x**a, 1e-12)
        result = safe_base ** (b - 1)
        return a * b * x ** (a - 1) * result
    return (f_R_ks,)


@app.cell
def _(gamma):
    def F_R_gamma(x, alpha, theta): # CDF of Gamma
        return gamma.cdf(x, a=alpha, scale=theta)
    return (F_R_gamma,)


@app.cell
def _(gamma):
    def f_R_gamma(x, alpha, theta): # PDF of Gamma
        return gamma.pdf(x, a=alpha, scale=theta)
    return (f_R_gamma,)


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
def _(expon):
    def F_T_Exp(x, lam2):
        return expon.cdf(x, scale=1/lam2)
    return


@app.cell
def _(expon):
    def f_T_Exp(x, lam2):
        return expon.pdf(x, scale=1/lam2)
    return (f_T_Exp,)


@app.cell
def _(weibull_min):
    def F_T_weibull(x, c):
        return weibull_min.cdf(x, c)
    return


@app.cell
def _(weibull_min):
    def f_T_weibull(x, c):
        return weibull_min.pdf(x, c)
    return (f_T_weibull,)


@app.cell
def _(norm):
    def F_T_Norm(x, mu, sigma):
        return norm.cdf(x, loc=mu, scale=sigma)
    return (F_T_Norm,)


@app.cell
def _(norm):
    def f_T_Norm(x, mu, sigma):
        return norm.pdf(x, loc=mu, scale=sigma)
    return (f_T_Norm,)


@app.cell
def _(mo):
    mo.md(r"""## Composite""")
    return


@app.cell
def _(Q_Y, f_R_ks, f_T_weibull, uQ_Y):
    def w_kc(x, a, b, lam, c):
        return f_T_weibull(Q_Y(F_R_ks(x, a, b), lam), c) * uQ_Y(F_R_ks(x, a, b), lam) * f_R_ks(x, a, b)
    return (w_kc,)


@app.cell
def _(Q_Y, f_R_ks, f_T_Exp, uQ_Y):
    def e_kc(x, a, b, lam, lam2):
        return f_T_Exp(Q_Y(F_R_ks(x, a, b), lam), lam2) * uQ_Y(F_R_ks(x, a, b), lam) * f_R_ks(x, a, b)
    return (e_kc,)


@app.cell
def _(Q_Y, f_R_ks, f_T_Norm, uQ_Y):
    def n_kc(x, a, b, lam, mu, sigma):
        return f_T_Norm(Q_Y(F_R_ks(x, a, b), lam), mu, sigma) * uQ_Y(F_R_ks(x, a, b), lam) * f_R_ks(x, a, b)
    return (n_kc,)


@app.cell
def _(F_R_gamma, Q_Y, f_R_gamma, f_T_Norm, uQ_Y):
    def n_gc(x, alpha, theta, mu, sigma, lam): # Normal - Gamma {Cauchy}
        return f_T_Norm(Q_Y(F_R_gamma(x, alpha, theta), lam), mu, sigma) * uQ_Y(F_R_gamma(x, alpha, theta), lam) * f_R_gamma(x, alpha, theta)
    return (n_gc,)


@app.cell
def _(F_R_gamma, F_T_Norm, Q_Y):
    def N_GC(x, alpha, theta, mu, sigma, lam): # Normal - Gamma {Cauchy}
        return F_T_Norm(Q_Y(F_R_gamma(x, alpha, theta), lam), mu, sigma)
    return


@app.cell
def _(mo):
    mo.md(r"""## Calculations""")
    return


@app.cell
def _(differential_evolution, np, w_kc):
    def calc_params_WKC(data):
        # Log-likelihood function with zero protection
        def ll(params, data):
            a, b, lam, c = params
            wkc_vals = w_kc(data, a, b, lam, c)
            wkc_vals[wkc_vals <= 0] = 1e-12
            return -np.sum(np.log(wkc_vals))

        # Bounds (tune if needed)
        bounds = [
            (1e-12, 2),    # a
            (1e-12, 1),    # b
            (1e-12, 2),    # lam
            (1e-12, 5),    # c
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
            tol=1e-7,                # Convergence tolerance (lower = more precise)
            updating="immediate",   # Update as soon as better solution is found (more aggressive)
            )

        print("Best log-likelihood:", -result.fun)
        print("Best parameters:", result.x)
        return result.x
    return (calc_params_WKC,)


@app.cell
def _(differential_evolution, e_kc, np):
    def calc_params_EKC(data):
        # Log-likelihood function with zero protection
        def ll(params, data):
            a, b, lam, lam2 = params
            ekc_vals = e_kc(data, a, b, lam, lam2)
            ekc_vals[ekc_vals <= 0] = 1e-12
            return -np.sum(np.log(ekc_vals))

        # Bounds (tune if needed)
        bounds = [
            (1e-5, 100),    # a
            (1e-5, 100),    # b
            (1e-5, 100),    # lam
            (1e-5, 100),    # lam2
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
            tol=1e-7,                # Convergence tolerance (lower = more precise)
            updating="immediate",   # Update as soon as better solution is found (more aggressive)
            )

        print("Best log-likelihood:", -result.fun)
        print("Best parameters:", result.x)
        return result.x
    return (calc_params_EKC,)


@app.cell
def _(differential_evolution, n_kc, np):
    def calc_params_NKC(data):
        # Log-likelihood function with zero protection
        def ll(params, data):
            a, b, lam, mu, sigma = params
            nkc_vals = n_kc(data, a, b, lam, mu, sigma)
            nkc_vals[nkc_vals <= 0] = 1e-12
            return -np.sum(np.log(nkc_vals))

        # Bounds (tune if needed)
        bounds = [
            (1e-5, 100),    # a
            (1e-5, 100),    # b
            (1e-5, 100),    # lam
            (-100, 100),    # mu
            (1e-5, 100)     # sigma
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
            tol=1e-7,                # Convergence tolerance (lower = more precise)
            updating="immediate",   # Update as soon as better solution is found (more aggressive)
            )

        print("Best log-likelihood:", -result.fun)
        print("Best parameters:", result.x)
        return result.x
    return


@app.cell
def _(differential_evolution, n_gc, np):
    def calc_params_NGC(data):
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
        result = differential_evolution(
            lambda p: ll(p, data),
            bounds,
            strategy="best1bin",     # Good default mutation strategy
            maxiter=1000,            # 🚀 Increase number of generations (default is 100)
            popsize=20,              # 🚀 Increase number of individuals per generation (default is 15)
            mutation=(0.5, 1.0),     # Mutation strength (can widen range to explore more)
            recombination=0.7,       # Crossover probability (how much mixing between solutions)
            polish=True,             # Fine-tune best solution with L-BFGS-B at the end
            tol=1e-7,                # Convergence tolerance (lower = more precise)
            updating="immediate",   # Update as soon as better solution is found (more aggressive)
            )

        print("Best log-likelihood:", -result.fun)
        print("Best parameters:", result.x)
    return


@app.cell
def _(datag, differential_evolution, gamma, np):
    def calc_params_gamma(data):
        # Custom Gamma log-likelihood (protect against log(0))
        def gamma_ll(params, datag):
            alpha, theta = params
            pdf_vals = gamma.pdf(datag, a=alpha, scale=theta)
            pdf_vals[pdf_vals <= 0] = 1e-12
            return -np.sum(np.log(pdf_vals))

        # Bounds for alpha and theta
        gamma_bounds = [
            (1e-5, 20),   # alpha (shape)
            (1e-5, 20),   # theta (scale)
        ]

        # Run optimization for Gamma distribution
        gamma_result = differential_evolution(
            lambda p: gamma_ll(p, datag),
            bounds=gamma_bounds,
            strategy="best1bin",
            maxiter=1000,
            popsize=20,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=True,
            tol=1e-7,
            updating="immediate"
        )

        # Print results
        print("\n=== Custom Gamma Fit ===")
        print("Best log-likelihood:", -gamma_result.fun)
        print("Best parameters (alpha, theta):", gamma_result.x)
    return


@app.cell
def _(calc_params_EKC, calc_params_WKC, norm, np, skewnorm):
    x = np.linspace(1e-12,1 - 1e-12, 100000)

    norm1 = np.clip(norm.rvs(loc=0.5, scale=0.1, size=1000), 1e-12, 1 - 1e-12)
    norm2 = np.clip(norm.rvs(loc=0.5, scale=0.2, size=1000), 1e-12, 1 - 1e-12)
    rskew = np.clip(skewnorm.rvs(a=10, loc=0.2, scale=0.15, size=1000), 1e-12, 1 - 1e-12)
    lskew = np.clip(skewnorm.rvs(a=-10, loc=0.8, scale=0.15, size=1000), 1e-12, 1 - 1e-12)
    bimod = np.clip(np.concatenate([
        norm.rvs(loc=0.3, scale=0.05, size=500),
        norm.rvs(loc=0.7, scale=0.05, size=500)
    ]), 1e-12, 1 - 1e-12)

    norm1_params = calc_params_WKC(norm1)
    norm2_params = calc_params_EKC(norm2)
    rskew_params = calc_params_WKC(rskew)
    lskew_params = calc_params_WKC(lskew)
    bimod_params = calc_params_WKC(bimod)
    return (
        bimod_params,
        lskew_params,
        norm1_params,
        norm2,
        norm2_params,
        rskew_params,
        x,
    )


@app.cell
def _(mo):
    mo_a = mo.ui.slider(start=0.001, stop=2, step=0.001, value=0.001, label=fr'$a$')
    mo_b = mo.ui.slider(start=0.001, stop=10, step=0.001, value=0.001, label=fr'$b$')
    #mo_alpha = mo.ui.slider(start=0, stop=20, step=0.01, value=2, label=fr'$\alpha$')
    #mo_theta = mo.ui.slider(start=0, stop=3, step=0.0001, value=3, label=fr'$\theta$')
    mo_lam = mo.ui.slider(start=0.1, stop=2, step=0.1, value=0.1, label=fr'$\lambda$')
    #mo_lam2 = mo.ui.slider(start=0.1, stop=100, step=0.1, value=0.1, label=fr'$\lambda2$')
    #mo_mu    = mo.ui.slider(start=-100, stop=100, step=1, value=0, label=fr'$\mu$')
    #mo_sigma    = mo.ui.slider(start=0.1, stop=75, step=1, value=1, label=fr'$\sigma$')
    mo_c = mo.ui.slider(start=0, stop=10, step=0.001, value=0.001, label=fr'$c$')


    mo.hstack([mo_a, mo_b, mo_lam, mo_c], justify='center', align='center', gap=5)
    return


@app.cell
def _(
    bimod_params,
    e_kc,
    lskew_params,
    norm1_params,
    norm2,
    norm2_params,
    plt,
    rskew_params,
    sns,
    w_kc,
    x,
):
    redc = '#a4031f'
    grayc = '#dddde3'
    greenc = '#88bf9b'
    bluec = '#3c91e6'
    pinkc = '#ffa0ac'

    fig, ax = plt.subplots()

    ax.plot(x, w_kc(x, *norm1_params), label='norm1', color=redc, lw=2)
    ax.plot(x, e_kc(x, *norm2_params), label='norm2', color=grayc, lw=2)
    ax.plot(x, w_kc(x, *rskew_params),  label='rskew',  color=greenc, lw=2)
    ax.plot(x, w_kc(x, *lskew_params), label='lskew',  color=bluec, lw=2)
    ax.plot(x, w_kc(x, *bimod_params), label='bimod',  color=pinkc, lw=2)

    sns.histplot(norm2, bins=70, stat='density', ax=ax, color=grayc)

    #ax.plot(x, w_kc(x, mo_a.value, mo_b.value, mo_lam.value, mo_c.value), lw=3)

    title = r'$\,W\!-\!K\{C\}\,$ Distribution'
    ax.set_title(title, fontsize=28, pad=20)
    ax.set_xlabel(xlabel=r'$X$ Value', fontsize=28, labelpad=20)
    ax.set_ylabel('Probability', fontsize=28, labelpad=20)

    ax.legend(fontsize=14)
    #ax.set_xlim(0,1)
    ax.set_ylim(0,6)
    ax.grid()
    ax.tick_params(labelsize=20)
    fig.gca()
    return


@app.cell
def _():
    #sns.histplot(data, bins=70, stat='density', ax=ax, color=grayc)

    #ax.plot(x, n_gc(x, s_alpha, s_theta, s_mu, s_sigma, s_lam), c=redc, label='Slider')
    #ax.plot(x, n_gc(x, alpha, theta, mu, sigma, lam), c=greenc, label='N-GC')
    #ax.plot(x, y_gamma1, c=redc, label=fr'$\alpha=20,\ \beta=0.5$')
    #ax.plot(x, y_gamma2, c=greenc, label=fr'$\alpha=3.5,\ \beta=1$')
    #ax.plot(x, y_gamma3, c=bluec, label=fr'$\alpha=20,\ \beta=0.8$')

    #ngc_norm1 = n_gc(x, 0.93, 10, 15.25, 3.95, 27.4)
    #ngc_norm2 = n_gc(x, 0.91, 10, 19.9, 10.5, 34)
    #ngc_lskew = n_gc(x, 6.5, 1.9, 17, 10, 9)
    #ngc_rskew = n_gc(x, 0.6, 10, 0, 18, 26)
    #ngc_bimodal = n_gc(x, 4.6, 2, 8, 20, 4.67)

    #ax.plot(x, ngc_norm1, c=redc, lw=2, label=fr'$\alpha=0.93,\ \theta=10,\ \mu=15.25,\ \sigma=3.95,\ \lambda=27.4$')
    #ax.plot(x, ngc_norm2, c=bluec, lw=2, label=fr'$\alpha=0.91,\ \theta=10,\ \mu=19.9,\ \sigma=10.5,\ \lambda=34$')
    #ax.plot(x, ngc_lskew, c=greenc, lw=2, label=fr'$\alpha=6.5,\ \theta=1.9,\ \mu=17,\ \sigma=10,\ \lambda=9$')
    #ax.plot(x, ngc_rskew, c=grayc, lw=2, label=fr'$\alpha=0.6,\ \theta=10,\ \mu=0,\ \sigma=18,\ \lambda=26$')
    #ax.plot(x, ngc_bimodal, c=pinkc, lw=2, label=fr'$\alpha=4.6,\ \theta=2,\ \mu=8,\ \sigma=20,\ \lambda=4.67$')
    return


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
