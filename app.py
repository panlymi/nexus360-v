import streamlit as st
import pandas as pd
import numpy as np

import streamlit as st

# Title and introduction
st.title("NexusRank360: Big Data-Driven MOORA System for Advanced Ranking")
st.markdown("""
This app evaluates and ranks alternatives using the **MOORA (Multi-Objective Optimization by Ratio Analysis) method**.
MOORA is applied to rank decision alternatives based on multiple criteria.
""")

# Provide a step-by-step explanation of the MOORA method
st.subheader("MOORA Method: Step-by-Step Explanation")

st.markdown("""
1. **Problem Definition**:
   - Identify the decision alternatives and the criteria on which they will be evaluated.
   - Alternatives could be different entities, such as companies or projects, and criteria could be factors like cost, performance, etc.

2. **Construct the Decision Matrix**:
   - Create a decision matrix where rows represent alternatives, and columns represent criteria. Each cell in the matrix represents how an alternative performs on a given criterion.

3. **Normalize the Decision Matrix**:
   - Normalize each criterion to make sure all criteria contribute equally, regardless of their units of measurement. 
     - **For Benefit Criteria**: Normalize by dividing each value by the Euclidean norm.
     - **For Cost Criteria**: Normalize by dividing the Euclidean norm by each value.

4. **Apply Weights**:
   - Assign weights to the criteria based on their importance. Multiply each normalized value by its corresponding weight to get the weighted normalized matrix.

5. **Calculate the MOORA Score**:
   - The MOORA Score for each alternative is calculated as the sum of the weighted normalized values across all criteria. Higher MOORA Scores indicate better-performing alternatives.

6. **Rank the Alternatives**:
   - Rank the alternatives based on their MOORA Scores. The alternative with the highest score is considered the best.

7. **Sensitivity Analysis** (Optional):
   - Perform sensitivity analysis to see how changes in criteria weights or values affect the rankings. This can help assess the stability and reliability of the rankings.
""")

# --- Main Application UI ---
st.title("NexusRank360: Big Data-Driven MOORA System")
st.markdown("""
This app evaluates and ranks alternatives using the **MOORA (Multi-Objective Optimization by Ratio Analysis) method**.
Upload your data to begin.
""")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data from the uploaded file
    try:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        st.write("### Data Preview:")
        st.dataframe(df_input.head())

        # --- User Configuration using columns for a clean layout ---
        st.sidebar.header("Configuration")
        
        # FIX 1: Explicitly separate Alternative from Criteria columns
        alt_col = st.sidebar.selectbox(
            "1. Select your 'Alternative' column (non-numeric)",
            options=df_input.columns
        )
        
        criteria_cols = st.sidebar.multiselect(
            "2. Select your 'Criteria' columns (must be numeric)",
            options=[col for col in df_input.columns if col != alt_col and pd.api.types.is_numeric_dtype(df_input[col])]
        )

        if criteria_cols:
            # Create a clean DataFrame with only the numeric criteria
            df_criteria = df_input.set_index(alt_col)[criteria_cols]

            st.sidebar.subheader("3. Set Weights and Impacts")
            weights = {}
            impacts = {}
            
            for col in criteria_cols:
                st.sidebar.markdown(f"**{col}**")
                weights[col] = st.sidebar.slider(f"Weight", 0.0, 1.0, round(1/len(criteria_cols), 2), 0.01, key=f"weight_{col}")
                impacts[col] = st.sidebar.radio(f"Impact", ["Benefit", "Cost"], key=f"impact_{col}", horizontal=True)
                st.sidebar.markdown("---")

            if st.sidebar.button("🚀 Calculate Ranks", type="primary", use_container_width=True):
                # Perform calculation and display results
                st.write("### MOORA Ranking Results")
                
                final_ranking = moora_method(df_criteria, weights, impacts)
                
                # Display results with gradient coloring
                st.dataframe(
                    final_ranking.style.format({'MOORA Score': "{:.4f}"})
                                      .background_gradient(cmap='viridis_r', subset=['Rank']),
                    use_container_width=True
                )
                
                # Plot the ranking
                st.write("### Ranking Visualization")
                st.bar_chart(final_ranking[['MOORA Score']])

        else:
            st.info("Please select your criteria columns in the sidebar to proceed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure your file format is correct and you have selected the right columns.")
else:
    st.warning("Please upload a file to start the analysis.")
