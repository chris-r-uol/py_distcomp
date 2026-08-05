import streamlit as st
import quantile_multi_comparison as qmc
import empirical_plots as ep
import gofstat as gf
import off_model as om
import off_model_plots as omp
import numpy as np
import pandas as pd

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
            options=["normal", "gamma", "exponential", "uniform"],
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
        
        st.session_state.data = data
        
    
    
    data = st.session_state.get('data', None)
    
    
    if data is not None:
        with st.expander("Show Data"):
            st.dataframe(data, use_container_width=True)
        st.header('Empirical Data')
        tabs2 = st.tabs(['Empirical Density Plot', 'Empirical CDF Plot'])
        with tabs2[0]:
            st.plotly_chart(ep.empirical_density_plot(data), use_container_width=True)
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
        cf_fig = qmc.cullen_and_frey_plot(data, seed=42)
        st.plotly_chart(cf_fig, use_container_width=True)

        st.header("Distribution Comparison Plots")
        which_distributions = st.multiselect("Select Distributions", options=qmc.SUPPORTED_DISTRIBUTIONS.keys(), default=['normal'])

        if which_distributions:
            qq_fig = qmc.quantile_comparison_plot(data, which_distributions)

            tabs = st.tabs(['Q-Q Plot', 'Histogram Overlay', 'P-P Plot', 'CDF Comparison'])
            for i, fig in enumerate(qq_fig):
                with tabs[i]:
                    st.plotly_chart(fig, use_container_width=True)

            st.header("Goodness of Fit")
            st.caption("Equivalent to R's gofstat(). Lower statistics, AIC and BIC indicate a better fit.")
            try:
                fits = gf.fit_distributions(data, which_distributions)
                st.dataframe(gf.gofstat(fits), use_container_width=True)
            except ValueError as exc:
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
                "Distribution", options=list(qmc.SUPPORTED_DISTRIBUTIONS.keys()),
                index=list(qmc.SUPPORTED_DISTRIBUTIONS.keys()).index('gumbel'),
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
    else:
        st.warning("Please generate data first!")

    

if __name__ == "__main__":
    main()