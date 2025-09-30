import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import numpy as np
    import marimo as mo
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from scipy.constants import pi
    from scipy.stats import norm, cauchy, weibull_min, fisk

    EPSILON = 1e-3
    sns.set_theme(style="white",context="talk",font_scale=1.25,rc={"axes.edgecolor": "black", "grid.color": "silver"})
    plt.rcParams["figure.figsize"] = (10, 10)
    plt.rcParams["figure.dpi"] = 140
    plt.rcParams["lines.linewidth"] = 1.5
    plt.rcParams["axes.titlesize"] = 26
    plt.rcParams["axes.titlepad"] = 15
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
    return EPSILON, cauchy, mo, norm, np, pi, plt


@app.cell
def _(EPSILON, cauchy, norm, np, pi):
    def n_kc(x, a, b, gamma, mu):
        return (
            norm.pdf(cauchy.ppf((1 - np.power(1 - np.power(x, a), b)), scale=gamma), loc=mu)
            * (gamma * (1 / np.power(np.cos(pi * ((1 - np.power(1 - np.power(x, a), b)) - 0.5)), 2)) * pi)
            * (a * b * np.power(x, a - 1) * np.power(np.maximum((1.0 - np.power(x, a)), EPSILON), b - 1))
               )
    return (n_kc,)


@app.cell
def _(EPSILON, mo, n_kc, np, plt):
    def n_kc_fig(a, b, gamma, mu):
        x = np.linspace(EPSILON, 1 - EPSILON, 10000)
        fig, ax = plt.subplots()
        ax.plot(x, n_kc(x, a, b, gamma, mu))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 5.6)
        ax.set_xlabel(xlabel=r"$X$ Value")
        ax.set_ylabel("Density")
        ax.set_title(r'$\,N\!-\!K\{C\}\,$ PDF')
        ax.grid()
        return fig

    a = mo.ui.slider(start=0.5, stop=2.5, step=0.10, value=1, label='$a$', show_value=True)
    b = mo.ui.slider(start=0.5, stop=2.5, step=0.10, value=1, label='$b$', show_value=True)
    gamma = mo.ui.slider(start=0.5, stop=4.5, step=0.10, value=2, label=r'$\gamma$', show_value=True)
    mu = mo.ui.slider(start=-1.5, stop=1.5, step=0.10, value=0, label=r'$\mu$', show_value=True)
    return a, b, gamma, mu, n_kc_fig


@app.cell
def _(a, b, gamma, mo, mu, n_kc_fig):
    mo.vstack([
        mo.md("## Parameters :"),
        mo.md("#"),
        mo.hstack([a, b, gamma, mu],
            justify='center', align='center', gap=5),
        mo.as_html(n_kc_fig(a.value, b.value, gamma.value, mu.value))
        ])
    return


if __name__ == "__main__":
    app.run()
