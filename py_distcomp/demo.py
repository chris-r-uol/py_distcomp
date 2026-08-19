"""
Streamlit demo for py_distcomp.

Run it with ``py-distcomp-demo``, or with ``streamlit run app.py`` from a
checkout of the repository.
"""

import numpy as np
import pandas as pd
import streamlit as st

from py_distcomp import distributions as dists
from py_distcomp import estimation as est
from py_distcomp import empirical_plots as ep
from py_distcomp import bootdist as bd
from py_distcomp import bootdist_plots as bdp
from py_distcomp import gofstat as gf
from py_distcomp import mixture as mx
from py_distcomp import mixture_plots as mxp
from py_distcomp import off_model as om
from py_distcomp import off_model_plots as omp
from py_distcomp import quantile_multi_comparison as qmc

def main():
    st.title("Demonstrator App for Py Dist Comp")
    st.write("This app demonstrates the functionalities of the Py Dist Comp library.")
    
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    # Streamlit form for data generation
    with st.form("data_generation_form"):
        st.subheader("Data Generation Parameters")
        
        # Number of data points
        n_points = st.number_input(
            "Number of data points", 
            min_value=10, 
            max_value=10000, 
            value=50, 
            step=10
        )
        
        # Distribution selection
        distribution_type = st.selectbox(
            "Select distribution type",
            options=["normal", "gamma", "exponential", "uniform", "poisson", "negative binomial"],
            index=0
        )
        
        # Distribution parameters based on selection
        if distribution_type == "normal":
            loc = st.number_input("Location (mean)", value=0.0, step=0.1)
            scale = st.number_input("Scale (std dev)", value=1.0, min_value=0.1, step=0.1)
        elif distribution_type == "gamma":
            shape = st.number_input("Shape", value=2.0, min_value=0.1, step=0.1)
            scale = st.number_input("Scale", value=2.0, min_value=0.1, step=0.1)
        elif distribution_type == "exponential":
            scale = st.number_input("Scale", value=1.0, min_value=0.1, step=0.1)
        elif distribution_type == "uniform":
            low = st.number_input("Low", value=0.0, step=0.1)
            high = st.number_input("High", value=1.0, step=0.1)
        elif distribution_type == "poisson":
            lam = st.number_input("Lambda", value=3.0, min_value=0.1, step=0.1)
        elif distribution_type == "negative binomial":
            nb_size = st.number_input("Size", value=4.0, min_value=0.1, step=0.1)
            nb_mu = st.number_input("Mu (mean)", value=6.0, min_value=0.1, step=0.1)
        
        # Form submit button
        submitted = st.form_submit_button("Generate Data")
    
    # Generate data when form is submitted
    if submitted:
        if distribution_type == "normal":
            data = np.random.normal(loc=loc, scale=scale, size=n_points)
        elif distribution_type == "gamma":
            data = np.random.gamma(shape=shape, scale=scale, size=n_points)
        elif distribution_type == "exponential":
            data = np.random.exponential(scale=scale, size=n_points)
        elif distribution_type == "uniform":
            data = np.random.uniform(low=low, high=high, size=n_points)
        elif distribution_type == "poisson":
            data = np.random.poisson(lam=lam, size=n_points).astype(float)
        elif distribution_type == "negative binomial":
            data = np.random.negative_binomial(
                nb_size, nb_size / (nb_size + nb_mu), size=n_points
            ).astype(float)
        
        st.session_state.data = data
        st.session_state.is_discrete = distribution_type in ("poisson", "negative binomial")
        
    
    
    data = st.session_state.get('data', None)
    is_count = st.session_state.get('is_discrete', False)
    
    
    if data is not None:
        with st.expander("Show Data"):
            st.dataframe(data, use_container_width=True)
        st.header('Empirical Data')
        tabs2 = st.tabs(['Empirical Density Plot', 'Empirical CDF Plot'])
        with tabs2[0]:
            st.plotly_chart(
                ep.empirical_density_plot(data, discrete=is_count),
                use_container_width=True,
            )
        with tabs2[1]:
            c1, c2, c3 = st.columns(3)
            with c1:
                show_percentiles = st.checkbox("Show Percentiles", value=True)
            with c2:
                percentile_lines=None
                #percentile_lines = st.text_input(
                #    "Percentile Lines (comma-separated, e.g. 25,50,75)",
                #    value="25,50,75"
                #)
                #if percentile_lines:
                #    percentile_lines = [float(p.strip()) for p in percentile_lines.split(',')]
                #else:
                #    percentile_lines = None
            with c3:
                show_annotations = st.checkbox("Show Annotations", value=True)
            st.plotly_chart(ep.empirical_cdf_plot(data,show_percentiles=show_percentiles, percentile_lines=percentile_lines, show_annotations=show_annotations), use_container_width=True)
        
        st.header("Descriptive Statistics")
        st.caption("Equivalent to R's descdist(). Kurtosis is not excess kurtosis: a normal distribution gives 3.")
        summary = qmc.descdist(data)
        st.dataframe(
            pd.DataFrame([summary]).set_index('method'),
            use_container_width=True,
        )

        st.header("Cullen and Frey Plot")
        cf_fig = qmc.cullen_and_frey_plot(data, discrete=is_count, seed=42)
        st.plotly_chart(cf_fig, use_container_width=True)

        st.header("Distribution Comparison Plots")
        all_names = list(qmc.SUPPORTED_DISTRIBUTIONS.keys())
        discrete_names = [n for n in all_names if n in dists.DISCRETE_DISTRIBUTIONS]
        # Discrete and continuous families cannot be fitted to the same data.
        options = discrete_names if is_count else [n for n in all_names if n not in discrete_names]
        which_distributions = st.multiselect(
            "Select Distributions", options=options,
            default=[options[0]] if not is_count else ['poisson'],
        )

        if which_distributions:
            c1, c2, c3 = st.columns(3)
            with c1:
                fit_method = st.selectbox(
                    "Estimation method", options=list(est.ESTIMATION_METHODS),
                    help="mle: maximum likelihood; mme: moments; qme: quantiles; "
                         "mge: goodness-of-fit",
                )
            with c2:
                gof = st.selectbox(
                    "GOF statistic", options=list(est.GOF_STATISTICS),
                    disabled=fit_method != 'mge',
                ) if fit_method == 'mge' else None
            with c3:
                band = st.selectbox(
                    "Q-Q confidence band", options=["none", "95%", "99%"],
                )

            kwargs = {'gof': gof} if fit_method == 'mge' and gof else {}
            band_level = {"none": None, "95%": 0.95, "99%": 0.99}[band]
            try:
                fits = gf.fit_distributions(data, which_distributions,
                                            method=fit_method, **kwargs)
                params = [f.params for f in fits]
                qq_fig = qmc.quantile_comparison_plot(
                    data, which_distributions, dist_params=params,
                    confidence_band=band_level, band_niter=400, seed=42,
                )
            except ValueError as exc:
                st.warning(str(exc))
                qq_fig = qmc.quantile_comparison_plot(data, which_distributions)

            tabs = st.tabs(['Q-Q Plot', 'Histogram Overlay', 'P-P Plot', 'CDF Comparison'])
            for i, fig in enumerate(qq_fig):
                with tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)

            st.header("Parameter Uncertainty")
            st.caption(
                "Standard errors from the observed information, as R's fitdist reports them, "
                "beside bootstrap percentile intervals."
            )
            unc_dist = st.selectbox(
                "Distribution", options=which_distributions, key='unc_dist'
            )
            n_boot = st.slider("Bootstrap resamples", 100, 2000, 500, step=100)
            if st.button("Compute intervals"):
                try:
                    one = gf.fit_distributions(data, unc_dist,
                                               method=fit_method, **kwargs)[0]
                    with st.spinner(f"Refitting {n_boot} resamples..."):
                        boot = bd.bootdist(one, niter=n_boot, seed=42)
                except ValueError as exc:
                    st.warning(str(exc))
                else:
                    st.dataframe(boot.summary(), use_container_width=True)
                    if boot.n_failed:
                        st.warning(
                            f"{boot.n_failed} of {boot.niter} refits failed and were dropped."
                        )
                    st.plotly_chart(
                        bdp.bootdist_plot(boot), use_container_width=True
                    )

            st.header("Goodness of Fit")
            st.caption("Equivalent to R's gofstat(). Lower statistics, AIC and BIC indicate a better fit.")
            try:
                st.dataframe(gf.gofstat(fits), use_container_width=True)
            except (ValueError, NameError) as exc:
                st.warning(str(exc))
        else:
            st.info("Select at least one distribution to compare.")

        st.header("Off-Model Fraction")
        st.caption(
            "Rushton et al. (2021). Cuts the data at each percentile, refits, and picks "
            "the cut whose Q-Q relationship best follows the 1:1 line. The remainder is "
            "the off-model fraction."
        )
        c1, c2 = st.columns(2)
        with c1:
            om_model = st.selectbox(
                "Distribution", options=options,
                index=options.index('gumbel') if 'gumbel' in options else 0,
            )
        with c2:
            om_method = st.radio("R² definition", options=['identity', 'pearson'], horizontal=True)

        try:
            result = om.off_model_fraction(data, om_model, method=om_method)
        except ValueError as exc:
            st.warning(str(exc))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Off-model percentile", f"{result.percentile:g}")
            m2.metric("Off-model fraction", f"{result.fraction:g}%")
            m3.metric("R²", f"{result.r_squared:.4f}")
            st.dataframe(result.summary().to_frame('value'), use_container_width=True)

            om_tabs = st.tabs(['R² Sweep', 'Superposition', 'Cut Q-Q Grid'])
            with om_tabs[0]:
                st.plotly_chart(
                    omp.r_squared_sweep_plot({om_model: result}), use_container_width=True
                )
            with om_tabs[1]:
                st.plotly_chart(
                    omp.off_model_density_plot(result), use_container_width=True
                )
            with om_tabs[2]:
                st.plotly_chart(
                    omp.percentile_cut_qq_plot(data, om_model), use_container_width=True
                )

        st.header("Mixture Fit")
        st.caption(
            "Fits a superposition of distributions jointly by expectation-maximisation. "
            "Unlike the hard cut above, every observation gets a probability of belonging "
            "to each component."
        )
        options = list(qmc.SUPPORTED_DISTRIBUTIONS.keys())
        c1, c2 = st.columns(2)
        with c1:
            comp1 = st.selectbox("Component 1", options=options,
                                 index=options.index('gumbel'), key='mix1')
        with c2:
            comp2 = st.selectbox("Component 2", options=options,
                                 index=options.index('gumbel'), key='mix2')

        if st.button("Fit mixture"):
            try:
                with st.spinner("Running expectation-maximisation..."):
                    mix = mx.fit_mixture(data, (comp1, comp2))
            except ValueError as exc:
                st.warning(str(exc))
            else:
                if not mix.converged:
                    st.warning(
                        f"EM stopped at the iteration limit ({mix.n_iter}) without converging."
                    )
                m1, m2, m3 = st.columns(3)
                m1.metric("Component 2 weight", f"{mix.weights[1] * 100:.1f}%")
                m2.metric("AIC", f"{mix.aic:.1f}")
                m3.metric("EM iterations", mix.n_iter)
                st.dataframe(mix.summary().to_frame('value'), use_container_width=True)

                mx_tabs = st.tabs(['Density', 'Component Probability', 'vs Single Fits'])
                with mx_tabs[0]:
                    st.plotly_chart(mxp.mixture_density_plot(mix), use_container_width=True)
                with mx_tabs[1]:
                    st.plotly_chart(
                        mxp.component_probability_plot(mix), use_container_width=True
                    )
                with mx_tabs[2]:
                    singles = gf.fit_distributions(data, [comp1, comp2])
                    st.dataframe(gf.gofstat(singles + [mix]), use_container_width=True)
    else:
        st.warning("Please generate data first!")

    

if __name__ == "__main__":
    main()