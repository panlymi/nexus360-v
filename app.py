import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="Empowering Sustainable Futures", page_icon="🌐", layout="wide")

# --- Helper Function for MOORA (Correct & Standard Implementation) ---
def moora_method(df, weights, criteria_types):
    """
    Performs the standard MOORA method calculation and returns all intermediate steps.
    """
    # 1. Normalization (Vector Normalization)
    norm_df = df.copy()
    for col in df.columns:
        denominator = np.sqrt((df[col]**2).sum())
        if denominator == 0:
            norm_df[col] = 0
        else:
            norm_df[col] = df[col] / denominator
    
    # 2. Weighted Normalization
    weighted_df = norm_df.copy()
    for col in weighted_df.columns:
        weighted_df[col] *= weights[col]
        
    # 3. Calculate Performance Score (Yi)
    scores = []
    for i in range(len(weighted_df)):
        benefit_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Benefit')
        cost_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Cost')
        scores.append(benefit_score - cost_score)
        
    # 4. Create Final DataFrame with Scores and Ranks
    result_df = pd.DataFrame({'MOORA Score': scores}, index=df.index)
    result_df['Rank'] = result_df['MOORA Score'].rank(ascending=False).astype(int)
    
    return norm_df, weighted_df, result_df.sort_values(by='Rank')

# --- Main Application UI ---
st.title("Empowering Sustainable Futures with Big Data")
st.subheader("A Smart Framework for Ranking Complex Alternatives using MOORA")
st.markdown("""
This app evaluates and ranks alternatives using the **MOORA (Multi-Objective Optimization by Ratio Analysis) method**.
Upload your data to begin.
""")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

        st.write("### Data Preview:")
        st.dataframe(df_input.head())

        st.sidebar.header("Configuration")
        
        alt_col = st.sidebar.selectbox(
            "1. Select your 'Alternative' column (non-numeric)",
            options=df_input.columns
        )
        
        criteria_cols = st.sidebar.multiselect(
            "2. Select your 'Criteria' columns (must be numeric)",
            options=[col for col in df_input.columns if col != alt_col and pd.api.types.is_numeric_dtype(df_input[col])]
        )

        if criteria_cols:
            df_criteria = df_input.set_index(alt_col)[criteria_cols].dropna()

            st.sidebar.subheader("3. Set Weights and Impacts")
            weights = {}
            impacts = {}
            
            for col in criteria_cols:
                st.sidebar.markdown(f"**{col}**")
                weights[col] = st.sidebar.slider(f"Weight", 0.0, 1.0, round(1/len(criteria_cols), 2), 0.01, key=f"weight_{col}")
                impacts[col] = st.sidebar.radio(f"Impact", ["Benefit", "Cost"], key=f"impact_{col}", horizontal=True)
                st.sidebar.markdown("---")

            if st.sidebar.button("🚀 Calculate Ranks", type="primary", use_container_width=True):
                st.write("### MOORA Ranking Results")
                
                normalized_matrix, weighted_matrix, final_ranking = moora_method(df_criteria, weights, impacts)
                
                st.dataframe(
                    final_ranking.style.format({'MOORA Score': "{:.4f}"})
                                      .background_gradient(cmap='viridis_r', subset=['Rank']),
                    use_container_width=True
                )
                
                st.write("### Ranking Visualization")
                st.bar_chart(final_ranking[['MOORA Score']])
                
                st.markdown("---")

                with st.expander("View Step-by-Step Calculation Details"):
                    st.subheader("Step 1: Initial Decision Matrix (Your Data)")
                    st.write("This is the raw data used for the calculation, containing only the numeric criteria.")
                    st.dataframe(df_criteria)

                    st.subheader("Step 2: Normalized Decision Matrix")
                    ### NEW: Updated to st.markdown for proper symbol rendering ###
                    st.markdown("Each value ($x_{ij}$) is normalized to create a common scale ($n_{ij}$) using the vector normalization formula:")
                    st.latex(r'''
                        n_{ij} = \frac{x_{ij}}{\sqrt{\sum_{k=1}^{m} x_{kj}^2}}
                    ''')
                    st.write("Where *m* is the total number of alternatives.")
                    st.dataframe(normalized_matrix.style.format("{:.4f}"))

                    st.subheader("Step 3: Weighted Normalized Decision Matrix")
                    ### NEW: Updated to st.markdown for proper symbol rendering ###
                    st.markdown("The normalized values ($n_{ij}$) are multiplied by their corresponding criterion weight ($w_j$) to get the weighted value ($v_{ij}$):")
                    st.latex(r'''
                        v_{ij} = w_j \times n_{ij}
                    ''')
                    st.dataframe(weighted_matrix.style.format("{:.4f}"))
                    
                    st.subheader("Step 4: Final Score Calculation")
                    ### NEW: Updated to st.markdown for proper symbol rendering ###
                    st.markdown("The final MOORA score ($Y_i$) for each alternative is calculated by summing the weighted values for 'Benefit' criteria and subtracting the weighted values for 'Cost' criteria:")
                    st.latex(r'''
                        Y_i = \sum_{j=1}^{g} v_{ij} - \sum_{j=g+1}^{n} v_{ij}
                    ''')
                    st.write("Where *g* is the number of benefit criteria and *(n-g)* is the number of cost criteria.")
                    st.dataframe(final_ranking.style.format({'MOORA Score': "{:.4f}"}))

        else:
            st.info("Please select your criteria columns in the sidebar to proceed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure your file format is correct and you have selected the right columns.")
else:
    st.warning("Please upload a file to start the analysis.")
