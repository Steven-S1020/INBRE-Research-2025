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
    from scipy.special import erfinv, gammainc
    from scipy.special import gamma as gamma_fun
    from scipy.constants import pi
    from scipy.stats import gamma, norm, cauchy, expon
    from scipy.integrate import quad, trapezoid

    sns.set_theme(style='white', context='poster', font_scale=1.25, palette='pastel')

    plt.rcParams['figure.figsize'] = (15,13)
    plt.rcParams['figure.dpi'] = 85
    return (
        cauchy,
        erfinv,
        gamma,
        gammainc,
        mo,
        norm,
        np,
        pi,
        plt,
        sqrt,
        trapezoid,
    )


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
def _(norm):
    def f_Y(x, mu, sigma): # PDF of Normal
        return norm.pdf(x, loc=mu, scale=sigma)
    return


@app.cell
def _(norm):
    def Q_Y(x, mu, sigma): # Quantile of Normal
        return norm.ppf(x, loc=mu, scale=sigma)
    return (Q_Y,)


@app.cell
def _(norm):
    def uQ_Y(x, mu, sigma): # Derivative Quantile of Normal
        p = norm.ppf(x, loc=mu, scale=sigma)
        return 1/(norm.pdf(p, loc=mu, scale=sigma))
    return (uQ_Y,)


@app.cell
def _(erfinv, gammainc, np, pi, sqrt):
    def uQ_Y_custom(x, alpha, theta, mu, sigma): # Derivative Quantile of Normal - Hard Coded
        return sigma * sqrt(2 * pi) * np.exp((erfinv(2 * gammainc(alpha, x/theta) - 1))**2)
    return


@app.cell
def _(np):
    def f_T_exp(x, lam): # PDF of Exponential
        return 1 * np.exp(-lam * x)
    return (f_T_exp,)


@app.cell
def _(cauchy):
    def F_T(x, tau, delta): # CDF of Exponential
        return cauchy.cdf(x, loc=tau, scale=delta)
    return (F_T,)


@app.cell
def _(cauchy):
    def f_T(x, tau, delta): # PDF of Cauchy
        return cauchy.pdf(x, loc=tau, scale=delta)
    return (f_T,)


@app.cell
def _(F_R, F_T, Q_Y):
    def C_GN(x, alpha, theta, mu, sigma, tau, delta): # CDF of the Cauchy - Gamma {Normal}
        return F_T(Q_Y(F_R(x, alpha, theta), mu, sigma), tau, delta) 
    return


@app.cell
def _(F_R, Q_Y, f_R, f_T_exp, uQ_Y):
    def e_gn(x, alpha, theta, mu, sigma, lam): # PDF of the Exponential - Gamma {Normal}
        return f_T_exp(Q_Y(F_R(x, alpha, theta), mu, sigma), lam) * uQ_Y(F_R(x, alpha, theta), mu, sigma) * f_R(x, alpha, theta)
    return


@app.cell
def _(F_R, Q_Y, f_R, f_T, uQ_Y):
    def c_gn(x, alpha, theta, mu, sigma, tau, delta): # PDF of the Cauchy - Gamma {Normal}
        return f_T(Q_Y(F_R(x, alpha, theta), mu, sigma), tau, delta) * uQ_Y(F_R(x, alpha, theta), mu, sigma) * f_R(x, alpha, theta)
    return (c_gn,)


@app.cell
def _(mo):
    mo_alpha = mo.ui.number(start=1, stop=5, step=0.25, value=1.25, label=fr'$\alpha$')
    mo_theta = mo.ui.number(start=0.50, stop=3, step=0.50, value=0.50, label=fr'$\theta$')
    mo_mu    = mo.ui.number(start=-10, stop=10, step=0.25, value=1, label=fr'$\mu$')
    mo_sigma = mo.ui.number(start=-5, stop=5, step=0.25, value=1, label=fr'$\sigma$')
    mo_tau   = mo.ui.number(start=0.25, stop=5, step=0.25, value=1, label=fr'$\tau$')
    mo_delta = mo.ui.number(start=-5, stop=5, step=0.25, value=1, label=fr'$\delta$')
    mo_lam = mo.ui.number(start=0.10, stop=5, step=0.10, value=1, label=fr'$\lambda$')


    mo.hstack([mo_alpha, mo_theta, mo_mu, mo_sigma, mo_tau, mo_delta, mo_lam], justify='center', align='center', gap=5)
    return mo_alpha, mo_delta, mo_lam, mo_mu, mo_sigma, mo_tau, mo_theta


@app.cell
def _(mo_alpha, mo_delta, mo_lam, mo_mu, mo_sigma, mo_tau, mo_theta, np):
    alpha = mo_alpha.value
    theta = mo_theta.value
    mu    = mo_mu.value
    sigma = mo_sigma.value
    tau   = mo_tau.value
    delta = mo_delta.value
    lam = mo_lam.value

    x = np.linspace(0,20,1000)
    return alpha, delta, mu, sigma, tau, theta, x


@app.cell
def _(alpha, c_gn, delta, mu, sigma, tau, theta):
    c_gn(19.2,alpha,theta,mu,sigma,tau,delta)
    return


@app.cell
def _(alpha, c_gn, delta, mu, np, sigma, tau, theta, trapezoid):
    x_intervals = np.linspace(0.1,19.2,100)
    trapezoid(c_gn(x_intervals,alpha,theta,mu,sigma,tau,delta), x_intervals)
    return


@app.cell
def _(alpha, c_gn, delta, mu, plt, sigma, tau, theta, x):
    fig, ax = plt.subplots()




    ax.plot(x, c_gn(x, alpha, theta, mu, sigma, tau, delta))
    #ax.plot(x, e_gn(x, alpha, theta, mu, sigma, lam))
    #ax.plot(x, f_T_exp(Q_Y(F_R(x, alpha, theta), mu, sigma), lam) * f_R(x, alpha, theta) * uQ_Y(F_R(x, alpha, theta), mu, sigma))
    ax.grid()

    ax.tick_params(labelsize=20)
    fig.gca()
    return


if __name__ == "__main__":
    app.run()
