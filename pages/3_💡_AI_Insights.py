"""automated ai dataset insights page."""

import streamlit as st

st.title("💡 Automated Dataset Insights")

if "raw_df" not in st.session_state:
    st.warning("Please upload a dataset on the main Home page first!")
else:
    df = st.session_state.raw_df
    numeric_df = df.select_dtypes(include=["number"])

    if not numeric_df.empty and numeric_df.shape[1] > 1:
        corr_matrix = numeric_df.corr()
        st.markdown("#### 🔍 Correlation Highlights")
        found_corr = False
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                col_a = corr_matrix.columns[i]
                col_b = corr_matrix.columns[j]
                val = corr_matrix.iloc[i, j]
                if abs(val) > 0.6:
                    found_corr = True
                    direction = "positive" if val > 0 else "negative"
                    st.success(f"Strong **{direction} correlation** ({val:.2f}) found between **{col_a}** and **{col_b}**.")
        
        if not found_corr:
            st.info("No strong linear correlations (> 0.6 absolute value) detected among numeric columns.")
    else:
        st.warning("Not enough numeric columns available to generate automated statistical correlations.")