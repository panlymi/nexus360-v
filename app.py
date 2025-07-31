import streamlit as st
import pandas as pd
import numpy as np

# --- Page Configuration ---
st.set_page_config(page_title="NexusRank 360", page_icon="🌐", layout="wide")

# --- Helper Function for MOORA ---
def moora_method(df, weights, criteria_types):
    """
    Performs the standard MOORA method calculation and returns all intermediate steps.
    """
    # Normalization (Vector Normalization)
    norm_df = df.copy()
    for col in df.columns:
        denominator = np.sqrt((df[col]**2).sum())
        if denominator == 0:
            norm_df[col] = 0
        else:
            norm_df[col] = df[col] / denominator
    
    # Weighted Normalization
    weighted_df = norm_df.copy()
    for col in weighted_df.columns:
        weighted_df[col] *= weights[col]
        
    # Calculate Performance Score (Yi)
    scores = []
    for i in range(len(weighted_df)):
        benefit_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Benefit')
        cost_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Cost')
        scores.append(benefit_score - cost_score)
        
    # Create Final DataFrame with Scores and Ranks
    result_df = pd.DataFrame({'MOORA Score': scores}, index=df.index)
    result_df['Rank'] = result_df['MOORA Score'].rank(ascending=False).astype(int)
    
    return result_df.sort_values(by='Rank'), norm_df, weighted_df

# --- Main Application UI ---
st.title("NexusRank360: Big Data-Driven MOORA System")
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
            "1. Select your 'Alternative' column",
            options=df_input.columns
        )
        
        criteria_cols = st.sidebar.multiselect(
            "2. Select your 'Criteria' columns",
            options=[col for col in df_input.columns if col != alt_col and pd.api.types.is_numeric_dtype(df_input[col])]
        )

        if criteria_cols:
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
                final_ranking, norm_df, weighted_df = moora_method(df_criteria, weights, impacts)
                
                st.write("### MOORA Ranking Results")
                st.dataframe(
                    final_ranking.style.format({'MOORA Score': "{:.4f}"})
                                      .background_gradient(cmap='viridis_r', subset=['Rank']),
                    use_container_width=True
                )
                
                st.write("### Ranking Visualization")
                st.bar_chart(final_ranking[['MOORA Score']])

                # --- EXPANDER SECTION WITH CORRECTED STEP NUMBERS ---
                with st.expander("Show Detailed Calculation Steps (360° View)"):
                    # Step 1
                    st.subheader("Step 1: Initial Decision Matrix & Configuration")
                    st.markdown("This is the raw numeric data and user inputs used for the calculation.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("User-defined Weights:")
                        st.json(weights)
                    with col2:
                        st.write("User-defined Impacts:")
                        st.json(impacts)
                    
                    st.write("Decision Matrix:")
                    st.dataframe(df_criteria)
                    st.markdown("---")
                    
                    # Step 2
                    st.subheader("Step 2: Normalized Matrix")
                    st.markdown("Each value is normalized using the formula: `x / sqrt(sum(x^2))` for its column.")
                    st.dataframe(norm_df.style.format("{:.4f}"))
                    st.markdown("---")

                    # Step 3
                    st.subheader("Step 3: Weighted Normalized Matrix")
                    st.markdown("Each normalized value is multiplied by its corresponding criterion weight.")
                    st.dataframe(weighted_df.style.format("{:.4f}"))
                    st.markdown("---")

                    # Step 4
                    st.subheader("Step 4: Performance Scores & Final Rank")
                    st.markdown("The final score is `Sum(Benefit Criteria) - Sum(Cost Criteria)`. The ranks are based on these scores.")
                    st.dataframe(final_ranking.style.format({'MOORA Score': "{:.4f}"}))

        else:
            st.info("Please select your criteria columns in the sidebar to proceed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please ensure your file format is correct and you have selected the right columns.")
else:
    st.warning("Please upload a file to start the analysis.")
