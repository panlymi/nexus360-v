import streamlit as st
import pandas as pd
import numpy as np

# Display Step-by-Step Tables
st.subheader("Step-by-Step MOORA Process")

# Show Input Data Table
st.write("**Input Data**")
st.write(df)

# Show Normalized Data Table
st.write("**Normalized Data**")
st.write(normalized_df)

# Show Weighted Normalized Data Table
weighted_matrix = normalized_df * weights
df['Weighted Normalized'] = weighted_matrix.sum(axis=1)
st.write("**Weighted Normalized Data**")
st.write(weighted_matrix)

# Show MOORA Score Table
df['MOORA Score'] = weighted_matrix.sum(axis=1)
st.write("**MOORA Score**")
st.write(df[['Alternative', 'MOORA Score']])

# Final Ranking
df_sorted = df.sort_values(by='MOORA Score', ascending=False)
st.write("**Final Ranking**")
st.write(df_sorted[['Alternative', 'MOORA Score']])


# --- Page Configuration ---
st.set_page_config(page_title="NexusRank 360", page_icon="🌐", layout="wide")

# --- Helper Function for MOORA (Correct & Standard Implementation) ---
def moora_method(df, weights, criteria_types):
    """
    Performs the standard MOORA method calculation.
    """
    # 1. Normalization (Vector Normalization)
    norm_df = df.copy()
    for col in df.columns:
        norm_df[col] = df[col] / np.sqrt((df[col]**2).sum())
    
    # 2. Weighted Normalization
    weighted_df = norm_df.copy()
    for col in weighted_df.columns:
        weighted_df[col] *= weights[col]
        
    # 3. Calculate Performance Score (Yi)
    scores = []
    for i in range(len(weighted_df)):
        # Use 'Benefit' and 'Cost' terms as per the UI
        benefit_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Benefit')
        cost_score = sum(weighted_df.iloc[i][col] for col, c_type in criteria_types.items() if c_type == 'Cost')
        scores.append(benefit_score - cost_score)
        
    # 4. Create Final DataFrame with Scores and Ranks
    result_df = pd.DataFrame({'MOORA Score': scores}, index=df.index)
    result_df['Rank'] = result_df['MOORA Score'].rank(ascending=False).astype(int)
    return result_df.sort_values(by='Rank')

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
