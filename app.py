import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title and description of the app
st.title("NexusRank360: Big Data-Driven MOORA System for Advanced Ranking")
st.markdown("""
This app evaluates and ranks alternatives using the **MOORA (Multi-Objective Optimization by Ratio Analysis) method**.
MOORA is applied to rank decision alternatives based on multiple criteria.
""")

# Upload section for user to upload the dataset
uploaded_file = st.file_uploader("Upload your Excel or CSV file with the data", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load data from the uploaded file
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)

    # Display the first few rows of the dataset
    st.write("Data Preview:")
    st.write(df.head())

    # User inputs for weight and impact of each criterion
    st.sidebar.header("Input Criteria Weights")
    criteria = df.columns.tolist()
    weights = []
    impacts = []
    
    for col in criteria:
        weight = st.sidebar.slider(f"Weight for {col}", 0.0, 1.0, 0.1)
        weight = round(weight, 2)
        weights.append(weight)
        
        impact = st.sidebar.radio(f"Impact for {col} (Benefit or Cost)", options=["Benefit", "Cost"], key=f"{col}_impact")
        impacts.append(impact)

    # Normalize data with error handling for non-numeric values
    normalized_df = df.copy()
    for col in criteria:
        # Ensure the column is numeric
        normalized_df[col] = pd.to_numeric(df[col], errors='coerce')

        # Handle NaN values by filling them with zero (or any other strategy)
        normalized_df[col].fillna(0, inplace=True)
        
        # Compute the normalization: benefit or cost
        if impacts[criteria.index(col)] == "Benefit":
            # Normalize by dividing by the Euclidean norm of the column
            column_norm = np.sqrt((df[col] ** 2).sum())
            normalized_df[col] = df[col] / column_norm if column_norm != 0 else 0
        else:
            # Normalize by dividing the Euclidean norm of the column by each value
            column_norm = np.sqrt((df[col] ** 2).sum())
            normalized_df[col] = column_norm / df[col] if column_norm != 0 else 0
    
    # Apply the MOORA method (weighted sum)
    weighted_matrix = normalized_df * weights
    df['MOORA Score'] = weighted_matrix.sum(axis=1)

    # Sort the dataframe by MOORA Score
    df_sorted = df.sort_values(by='MOORA Score', ascending=False)

    # Display results
    st.write("MOORA Ranking Results:")
    st.write(df_sorted[['MOORA Score']])

    # Plot the ranking
    st.subheader("Ranking Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(df_sorted.index, df_sorted['MOORA Score'], color='skyblue')
    ax.set_xlabel('MOORA Score')
    ax.set_ylabel('Alternatives')
    ax.set_title('Ranking of Alternatives Based on MOORA Method')
    plt.tight_layout()
    st.pyplot(fig)

else:
    st.warning("Please upload a file to start the analysis.")
