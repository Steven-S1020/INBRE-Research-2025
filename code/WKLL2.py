import marimo

__generated_with = "0.13.15"
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

    epsilon = 1e-12


    #Thank you Steven Stokes!
    return epsilon, mo, np, plt


@app.cell
def _(epsilon, np):
    # Kumaraswamy PDF (robust)
    def fR(x, a, b):
        x = np.clip(x, epsilon, 1 - epsilon)
        fx = a * b * x**(a - 1) * (1 - x**a)**(b - 1)
        return np.maximum(fx, epsilon)

    # Kumaraswamy CDF (robust)
    def FR(x, a, b):
        x = np.clip(x, epsilon, 1 - epsilon)
        Fx = 1 - (1 - x**a)**b
        return np.clip(Fx, epsilon, 1 - epsilon)

    return FR, fR


@app.cell
def _(epsilon, np):
    # Log-Logistic Quantile Function (robust), with β = 1
    def QY(p, alpha):
        # Clip p to strictly within (0, 1)
        p = np.clip(p, epsilon, 1 - epsilon)
    
        # Compute quantile
        q = alpha * (p / (1 - p))
    
        # Avoid returning 0 or Inf
        return np.maximum(q, epsilon)

    # Log-Logistic PDF (robust), with β = 1
    def fY(y, alpha):
        # Ensure y is positive and clipped
        y = np.maximum(y, epsilon)
    
        num = 1 / alpha
        denom = (1 + (y / alpha))**2
    
        fy = num / denom
    
        # Avoid zero or infinite values
        return np.maximum(fy, epsilon)


    return QY, fY


@app.cell
def _(epsilon, np):
    # Weibull PDF with λ = 1
    def fT(y, k):
        y = np.maximum(y, epsilon)
        val = k * y**(k - 1) * np.exp(-y**k)
        return np.maximum(val, epsilon)

    # Weibull CDF with λ = 1
    def FT(y, k):
        y = np.maximum(y, epsilon)
        val = 1 - np.exp(-y**k)
        return np.clip(val, epsilon, 1 - epsilon)

    return FT, fT


@app.cell
def _(FR, FT, QY, epsilon, fR, fT, fY, np):
    def gX(x, a, b, alpha, k):
    
        p = FR(x, a, b)
        q = QY(p, alpha)

        fy = fY(q, alpha)
        ft = fT(q, k)
        fr = fR(x, a, b)

        val = ft * (1 / fy) * fr

        # Replace non-finite values with epsilon
        val = np.where(np.isfinite(val), val, epsilon)

        return val

    def GX(x, a, b, alpha, k):
        p = FR(x, a, b)         # Kumaraswamy CDF
        q = QY(p, alpha)        # Log-Logistic Quantile
        return FT(q, k)         # Weibull CDF

    return (gX,)


@app.cell
def _(gX, np, plt):
    # Define x_vals from 0 to 1 (same as R seq(0, 1, length.out = 1000))
    x_vals = np.linspace(0, 1, 1000)

    # Compute gX for different parameter sets
    gx_vals1 = gX(x_vals, 5.7, 15, 0.1, 0.4)
    gx_vals2 = gX(x_vals, 2.226, 1.66, 5, 1.32)
    gx_vals3 = gX(x_vals, 3.6300, 1.6397, 5, 1.26)
    gx_vals4 = gX(x_vals, 1, 1.2, 4.3, 1)
    gx_vals5 = gX(x_vals, 1.5, 1.6, 10, 1)
    gx_vals6 = gX(x_vals, 1.5, 1.3, 0.1, 0.9)


    # Plot first curve in blue
    plt.plot(x_vals, gx_vals1, lw=2, color='blue', label='Params set 1')

    # Plot all other curves in red
    plt.plot(x_vals, gx_vals2, lw=2, color='blue', label='Params set 2')
    plt.plot(x_vals, gx_vals3, lw=2, color='yellow', label='Params set 3')
    plt.plot(x_vals, gx_vals4, lw=2, color='green', label='Params set 4')
    plt.plot(x_vals, gx_vals5, lw=2, color='orange', label='Params set 5')
    plt.plot(x_vals, gx_vals6, lw=2, color='black', label='Params set 6')

    plt.title('Plot of gX(x) with fitted parameters')
    plt.xlabel('x')
    plt.ylabel('gX(x)')
    plt.legend()
    plt.show()

    return


@app.cell
def _(mo):
    # Assume gX is defined as:
    # def gX(x, a, b, alpha, k):

    # Create your widget values (for example with magicgui or magicwidgets, assuming mo.ui.number is something similar)
    a = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    b = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    alpha = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)
    k = mo.ui.number(start=0.1, stop=100, step=0.1, value=1.0)

    mo.hstack([a, b, alpha, k], justify='center', align='center', gap=5)

    #ax4.plot(x, gX(x, 6.3, 1.9, 1.1, 0.7, 0.9, 1.9), lw=3)
    #ax4.plot(x, gX(x, 3.5, 2.4, 1.3, 1,  0.3, 0.85), lw=3)

    return a, alpha, b, k


@app.cell
def _(a, alpha, b, gX, k, np, plt):

    x1 = np.linspace(1e-12, 1 - 1e-12, 100000)

    # Compute y1 with correct parameters
    y1 = gX(x1, a.value, b.value, alpha.value, k.value)

    fig, ax = plt.subplots()
    ax.plot(x1, y1, label="gX(x)")
    ax.set_title(f"a={a.value}, b={b.value}, alpha={alpha.value}, k={k.value}")
    ax.set_xlabel("x")
    ax.set_ylabel("gX(x)")
    ax.grid(True)
    ax.legend()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
