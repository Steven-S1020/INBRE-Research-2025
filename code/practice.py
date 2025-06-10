import marimo

__generated_with = "0.13.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _(mo):
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm, pareto

    mo.show_code()

    sns.set_theme(style='white', context='poster', font_scale=1.25, palette='pastel')

    plt.rcParams['figure.figsize'] = (15,13)
    plt.rcParams['figure.dpi'] = 85
    return norm, np, pareto, plt, sns


@app.cell
def _(mo):
    mo.md(
        r"""
    # Examples of using Python with Statistical Distributions:
    ---
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #
    ## Normal Distribution Parameters :
    #
    """
    )
    return


@app.cell
def _(mo):
    mu = mo.ui.number(start=-2, stop=2, step=0.1, label=fr'$\mu$')
    std_dev = mo.ui.number(start=1, stop=3, step=0.1, label=fr'$\sigma$')
    return mu, std_dev


@app.cell
def _(mo, mu, std_dev):
    mo.hstack([mu, std_dev], justify='center', align='center', gap=5)
    return


@app.cell
def _(mu, norm, np, plt, sns, std_dev):
    mu_value = mu.value
    std_dev_value = std_dev.value

    data_norm = np.random.normal(mu_value, std_dev_value, 1000)

    x_norm_min, x_norm_max = min(data_norm), max(data_norm)

    x_norm = np.linspace(x_norm_min, x_norm_max, 1000)
    y_norm = norm.pdf(x_norm, loc=mu_value, scale=std_dev_value)

    ax = sns.histplot(data_norm, bins=16, stat='density')
    plt.plot(x_norm, y_norm, color='red', linewidth=3, label='PDF')
    plt.xlim(-3, 3)
    plt.ylim(0,0.55)
    ax.set_title('Normal Distribution', fontsize=40, pad=20)
    ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
    ax.set_ylabel('Probability', fontsize=40, labelpad=20)

    plt.vlines(x = mu_value, ymin = 0, ymax = max(y_norm), colors = 'purple', linestyle='--')
    ax.annotate(fr'$\mu={mu_value}$', xy=(mu_value,max(y_norm)), xytext=(1.5,0.45), arrowprops=dict(arrowstyle="simple",
                                fc="0.6", ec="none",
                                connectionstyle="arc3,rad=0.3"))

    ax.annotate(fr'$\sigma={std_dev_value}$', xy=(4., 1.), xycoords='data',
                xytext=(4.5, -1), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="bar",
                                ec="k",
                                shrinkA=5, shrinkB=5))

    ax.annotate('', xy=(4., 1.), xycoords='data',
                xytext=(4.5, -1), textcoords='data',
                arrowprops=dict(arrowstyle="<->",
                                connectionstyle="bar",
                                ec="k",
                                shrinkA=5, shrinkB=5))

    # Adjust tick label size
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.grid()
    plt.legend()

    # mo.show_code(plt.gcf())
    return (x_norm,)


@app.cell
def _(norm, plt, x_norm):
    y_norm_cdf = norm.cdf(x_norm)
    plt.xlim(-3,3)
    plt.grid()
    plt.xlabel('$X$ Values', fontsize=40, labelpad=20)
    plt.ylabel('Probability', fontsize=40, labelpad=20)
    plt.title('CDF of Normal Distribution', fontsize=40, pad=20)
    plt.plot(x_norm, y_norm_cdf, color='blue', label='CDF')
    plt.legend()
    plt.fill_between(x_norm,y_norm_cdf, alpha=0.6)
    plt.gcf()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #
    ---
    ## Using Numpy Normal Generator for Random Data :
    #
    """
    )
    return


@app.cell
def _(mo, np, pareto, plt, sns):
    data_pareto = np.random.pareto(3, 1000)

    x_pareto_min, x_pareto_max = min(data_pareto), 6

    x_pareto = np.linspace(x_pareto_min, x_pareto_max, 1000)
    y_pareto = pareto.pdf(x_pareto, 3, -1)

    sns.histplot(data_pareto, bins=75, stat='density')
    plt.plot(x_pareto, y_pareto, label='PDF', linewidth=3, color='red')
    plt.xlim(x_pareto_min, x_pareto_max)
    plt.title('Model Fitted with Pareto Distribution')
    plt.xlabel('$X$ Value')
    plt.grid()
    plt.legend()

    mo.show_code(plt.gcf())
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    #
    ---
    ## Next Section
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
