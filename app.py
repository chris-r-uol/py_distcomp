import streamlit as st
import quantile_multi_comparison as qmc
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
        
    
    which_distributions = st.multiselect("Select Distributions", options=qmc.SUPPORTED_DISTRIBUTIONS.keys(), default=['normal'])
    data = st.session_state.get('data', None)
    
    
    if data is not None:
        with st.expander("Show Data"):
            st.dataframe(data, use_container_width=True)
    
        qq_fig = qmc.quantile_comparison_plot(data, which_distributions)
        for fig in qq_fig:
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please generate data first!")

    

if __name__ == "__main__":
    main()