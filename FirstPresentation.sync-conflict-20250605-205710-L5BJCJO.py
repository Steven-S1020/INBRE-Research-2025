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
    from matplotlib.patches import FancyArrowPatch
    from scipy.stats import norm, gamma, skewnorm

    mo.show_code()

    sns.set_theme(style='white', context='poster', font_scale=1.25, palette='pastel')

    plt.rcParams['figure.figsize'] = (15,13)
    plt.rcParams['figure.dpi'] = 85
    return gamma, norm, np, plt, skewnorm, sns


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
    normal_mu = mo.ui.number(start=-2, stop=2, step=0.25, value=0, label=fr'$\mu$')
    normal_sigma = mo.ui.number(start=0.25, stop=3, step=0.25, value=1, label=fr'$\sigma$')
    return normal_mu, normal_sigma


@app.cell
def _(norm, np, plt, sns):
    def make_normal_fig(mu, sigma):

        data_norm = np.random.normal(mu, sigma, 1000)

        x_min_norm, x_max_norm = min(data_norm), max(data_norm)

        x_range_norm = np.linspace(x_min_norm, x_max_norm, 1000)
        y_pdf_norm = norm.pdf(x_range_norm, loc=mu, scale=sigma)

        fig, ax = plt.subplots()

        sns.histplot(data_norm, bins=16, stat='density', ax=ax)
        ax.plot(x_range_norm, y_pdf_norm, color='red', linewidth=3, label='PDF')
        ax.set_xlim(-3, 3)
        ax.set_ylim(0,0.55)
        ax.set_title('Normal Distribution', fontsize=40, pad=20)
        ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
        ax.set_ylabel('Probability', fontsize=40, labelpad=20)

        ax.vlines(x = mu, ymin = 0, ymax = max(y_pdf_norm), colors = 'purple', linestyle='--')
        ax.annotate(fr'$\mu={mu}$', xy=(mu,max(y_pdf_norm)), xytext=(1.5,0.45), arrowprops=dict(arrowstyle="simple",
                                    fc="0.6", ec="none",
                                    connectionstyle="arc3,rad=0.3"))

        # Adjust tick label size
        ax.tick_params(labelsize=20)

        ax.grid()
        ax.legend()

        return fig
    return (make_normal_fig,)


@app.cell
def _(make_normal_fig, mo, normal_mu, normal_sigma):
    def normal_tab():
        normal_outputs = [
            mo.md("---"),
            mo.md("#"),
            mo.md("### Parameters :"),
            mo.md("#"),
            mo.hstack([normal_mu, normal_sigma], justify='center', align='center', gap=5),
            mo.as_html(make_normal_fig(normal_mu.value, normal_sigma.value))
        ]

        return mo.vstack(normal_outputs)
    return (normal_tab,)


@app.cell
def _(mo):
    gamma_alpha = mo.ui.number(start=0.25, stop=999, step=1, value=4.75, label=fr'$\alpha$')
    gamma_beta = mo.ui.number(start=0.25, stop=10, step=1, value=4.5, label=fr'$\beta$')
    return gamma_alpha, gamma_beta


@app.cell
def _(gamma, np, plt, sns):
    def make_gamma_fig(alpha, beta, chi, exp):

        data = np.random.gamma(alpha, beta, 1000)

        x_min, x_max = min(data), max(data)

        x_range = np.linspace(x_min, x_max, 1000)
        y_pdf = gamma.pdf(x_range, a=alpha, scale=beta)

        fig, ax = plt.subplots()

        sns.histplot(data, bins=25, stat='density', ax=ax)
        ax.plot(x_range, y_pdf, color='red', linewidth=3, label='PDF')
        ax.set_title('Gamma Distribution', fontsize=40, pad=20)
        ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
        ax.set_ylabel('Probability', fontsize=40, labelpad=20)

        if chi == False and exp == False:
            ax.set_ylim(0,0.04)
            ax.set_xlim(25,225)
        elif chi == True:
            ax.set_xlim(850,1150)
        else:
            ax.set_xlim(x_min,x_max)

        # Adjust tick label size
        ax.tick_params(labelsize=20)

        ax.grid()
        ax.legend()
        return fig

    return (make_gamma_fig,)


@app.cell
def _(mo):
    gamma_radio = mo.ui.radio(options=["Custom","Chi-Square", "Exponential"], label="Parameter Options:")
    return (gamma_radio,)


@app.cell
def _(gamma_alpha, gamma_beta, gamma_radio, make_gamma_fig, mo):
    def gamma_tab():
        gamma_outputs = [
            mo.md("---"),
            mo.md("#"),
            mo.md("### Parameters :"),
            mo.md("#"),
            gamma_radio,
            mo.md(r"""
                - **Chi-Square**: Set $\alpha = \frac{\nu}{2}$, $\beta = 2$  
                - **Exponential**: Set $\alpha = 1$, $\beta = 1$
            """),
        ]
        gamma_alpha_value = 4.75
        gamma_beta_value = 4.5
        chi = False
        exp = False

        # Add controls only if needed
        if gamma_radio.value == 'Custom':
            gamma_outputs.append(
                mo.hstack([gamma_alpha, gamma_beta], justify='center', align='center', gap=5)
            )
            gamma_alpha_value = gamma_alpha.value
            gamma_beta_value = gamma_beta.value
        elif gamma_radio.value == 'Chi-Square':
            gamma_alpha_value = 499.5
            gamma_beta_value = 2
            chi = True
        elif gamma_radio.value == 'Exponential':
            gamma_alpha_value = 1
            gamma_beta_value = 1
            exp = True

        gamma_outputs.append(
            mo.as_html(make_gamma_fig(gamma_alpha_value, gamma_beta_value, chi, exp))
        )
        return mo.vstack(gamma_outputs)
    return (gamma_tab,)


@app.cell
def _(norm, np, plt, skewnorm, sns):
    def make_extra_figs(fig_num, mu_value, sigma_value, skew_value):
        if fig_num == 1:
            mu = mu_value
            sigma = sigma_value

            data_norm = np.random.normal(mu, sigma, 1000)

            x_min_norm, x_max_norm = min(data_norm), max(data_norm)

            x_range_norm = np.linspace(x_min_norm, x_max_norm, 1000)
            y_pdf_norm = norm.pdf(x_range_norm, loc=mu, scale=sigma)

            fig, ax = plt.subplots()

            sns.histplot(data_norm, bins=16, stat='density', ax=ax)
            ax.plot(x_range_norm, y_pdf_norm, color='red', linewidth=3, label='PDF')
            ax.set_xlim(-3, 3)
            ax.set_ylim(0,0.55)
            ax.set_title('Normal Distribution', fontsize=40, pad=20)
            ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
            ax.set_ylabel('Probability', fontsize=40, labelpad=20)

            ax.vlines(x = mu, ymin = 0, ymax = max(y_pdf_norm), colors = 'purple', linestyle='--')
            ax.annotate(fr'$\mu={mu}$', xy=(mu,max(y_pdf_norm)), xytext=(1.5,0.45), arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none", connectionstyle="arc3,rad=0.3"))

            ax.annotate('', xy=(mu+sigma,0.02), xytext=(mu-sigma,0.02), arrowprops=dict(arrowstyle="<->", lw=3, ec="gray"))

            ax.annotate(fr'$\sigma={sigma}$', xy=(mu,0.02), xytext=(1.75,0.25), arrowprops=dict(arrowstyle="simple", fc="0.6", ec="none", connectionstyle="arc3,rad=0.3"))

            # Adjust tick label size
            ax.tick_params(labelsize=20)

            ax.grid()
            ax.legend()
            return fig

        elif fig_num == 2:
            # data_norm = skewnorm.rvs(skew_value, loc=mu_value, scale=sigma_value, size=2000)
            # mu, sigma, skew, kurt = skewnorm.stats(skew_value, moments='mvsk')

            x_min_norm, x_max_norm = -3, 3

            x_range = np.linspace(x_min_norm, x_max_norm, 2000)
            y_pdf = skewnorm.pdf(x=x_range,a=skew_value,loc=mu_value,scale=sigma_value)
            y_pdf2 = skewnorm.pdf(x=x_range,a=-skew_value,loc=-mu_value,scale=sigma_value)

            fig, ax = plt.subplots()

            # sns.histplot(data_norm, bins=50, stat='density', ax=ax, kde=True)
            ax.plot(x_range, y_pdf, color='red', linewidth=3, label='PDF - Skewed Left')
            ax.fill_between(x_range,y_pdf, alpha=0.4,color='red')

            ax.plot(x_range, y_pdf2, color='lightblue', linewidth=3, label='PDF - Skewed Right')
            ax.fill_between(x_range,y_pdf2, alpha=0.5,color='lightblue')

            ax.set_xlim(-2, 2)
            ax.set_ylim(0,1)
            ax.set_title('Skewed Distributions', fontsize=40, pad=20)
            ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
            ax.set_ylabel('Probability', fontsize=40, labelpad=20)

            # Adjust tick label size
            ax.tick_params(labelsize=20)

            ax.grid()
            ax.legend()
            return fig

        elif fig_num == 3:
            x_min_norm, x_max_norm = -3, 3

            x_range = np.linspace(x_min_norm, x_max_norm, 2000)
            y_pdf = skewnorm.pdf(x=x_range,a=skew_value,loc=mu_value,scale=sigma_value)
            y_pdf2 = skewnorm.pdf(x=x_range,a=skew_value,loc=-mu_value,scale=sigma_value-0.5)


            fig, ax = plt.subplots()

            # sns.histplot(data_norm, bins=50, stat='density', ax=ax, kde=True)
            ax.plot(x_range, y_pdf, color='red', linewidth=3, label='PDF - Heavier Tail')
            ax.fill_between(x_range,y_pdf, alpha=0.4,color='red')

            ax.plot(x_range, y_pdf2, color='lightblue', linewidth=3, label='PDF - Lighter Tail')
            ax.fill_between(x_range,y_pdf2, alpha=0.5,color='lightblue')

            ax.set_xlim(-2, 2)
            ax.set_ylim(0,1)
            ax.set_title('Distributions Showing Kurtosis', fontsize=40, pad=20)
            ax.set_xlabel('$X$ Value', fontsize=40, labelpad=20)
            ax.set_ylabel('Probability', fontsize=40, labelpad=20)

            # Adjust tick label size
            ax.tick_params(labelsize=20)

            ax.grid()
            ax.legend()
            return fig
    return (make_extra_figs,)


@app.cell
def _(mo):
    extra_radio = mo.ui.radio(options=['mean_variance', 'skew', 'kurt'], label='Select Extra Fig to show:')
    return (extra_radio,)


@app.cell
def _(mo):
    extra_skew = mo.ui.number(start=-100, stop=100, step=1, value=0, label=fr'$skew$')
    return (extra_skew,)


@app.cell
def _(extra_radio, extra_skew, make_extra_figs, mo, normal_mu, normal_sigma):
    def extra_tab():
        extra_outputs = [
            mo.md('### Extra'),
            mo.md('---'),
            extra_radio
        ]

        if extra_radio.value == 'mean_variance':
            extra_outputs.append(
                mo.hstack([normal_mu, normal_sigma], justify='center', align='center', gap=5))
            extra_outputs.append(
                mo.as_html(make_extra_figs(1,normal_mu.value,normal_sigma.value,extra_skew.value)))
        elif extra_radio.value == 'skew':
            extra_outputs.append(
                mo.hstack([normal_mu, normal_sigma, extra_skew], justify='center', align='center', gap=5))
            extra_outputs.append(mo.as_html(make_extra_figs(3,normal_mu.value,normal_sigma.value,extra_skew.value)))

        elif extra_radio.value == 'kurt':
            extra_outputs.append(
                mo.hstack([normal_mu, normal_sigma], justify='center', align='center', gap=5))
            extra_outputs.append(
                mo.as_html(make_extra_figs(3,normal_mu.value,normal_sigma.value,extra_skew.value)))



        return mo.vstack(extra_outputs)
    return (extra_tab,)


@app.cell
def _(mo):
    mo.md(r"""## Visualization:""")
    return


@app.cell
def _(extra_tab, gamma_tab, mo, normal_tab):

    mo.ui.tabs({
        "Normal Distribution": normal_tab(),
        "Gamma Distribution": gamma_tab(),
        "Extra": extra_tab()
    })
    return


if __name__ == "__main__":
    app.run()
