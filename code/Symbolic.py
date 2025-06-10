import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import sympy as sp
    from sympy.stats import Gamma, Cauchy, Normal, cdf, density

    # Define variables
    x, alpha, theta = sp.symbols('x alpha theta', real=True, positive=True)
    p, mu, sigma, tau, delta = sp.symbols('p mu sigma tau delta', real=True)
    return (
        Cauchy,
        Gamma,
        Normal,
        alpha,
        cdf,
        delta,
        density,
        mu,
        p,
        sigma,
        sp,
        tau,
        theta,
        x,
    )


@app.cell
def _(
    Cauchy,
    Gamma,
    Normal,
    alpha,
    cdf,
    delta,
    density,
    mu,
    p,
    sigma,
    sp,
    tau,
    theta,
    x,
):
    T = Cauchy('T', tau, delta)
    R = Gamma('R', alpha, theta)
    Y = Normal('Y', mu, sigma)

    F_R = cdf(R)(x)

    f_T = density(T)(x)
    f_R = density(R)(x)
    f_Y = density(Y)(x)

    Q_Y = mu + sigma * sp.sqrt(2) * sp.erfinv(2 * p - 1)
    uQ_Y = sp.diff(Q_Y, p)

    Q_Y_FR = Q_Y.subs(p, F_R)
    uQ_Y_FR = uQ_Y.subs(p, F_R)

    f_T_QY_FR = f_T.subs(x, Q_Y_FR)

    C_GN = f_T_QY_FR * uQ_Y_FR * f_R

    C_GN_simp = C_GN.expand().simplify().doit()
    C_GN_simp_subbed = C_GN_simp.subs({
        alpha: 4,
        theta: 2,
        mu: 1,
        sigma: 1,
        tau: 1,
        delta: 1
    })
    return (C_GN_simp_subbed,)


@app.cell
def _(C_GN_simp_subbed, sp, x):
    sp.limit(C_GN_simp_subbed, x, sp.oo)
    return


if __name__ == "__main__":
    app.run()
