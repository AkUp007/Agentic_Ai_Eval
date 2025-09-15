import streamlit as st
import pandas as pd
import os
import time
import seaborn as sns
import matplotlib.pyplot as plt
from eval1 import evaluate_csv, score_with_gemini, score_with_groq, score_with_qwen

# -------------------------------
# Streamlit UI Setup
# -------------------------------

st.title("Agentic Evaluation Framework")

# File uploader for input CSV
uploaded_file = st.file_uploader("Upload CSV", type="csv")

# Slider for batch size selection (min=1, max=50, default=10)
batch_size = st.slider("Batch Size", 1, 50, 10)

# Input box for limiting rows (0 means process all rows)
row_limit = st.number_input("Row Limit (leave 0 for all)", min_value=0, value=0)

# model chice
model_choice = st.selectbox(
    "Choose Evaluation Model",
    ["gemini-2.5-flash", "llama-3.1-8b-instant", "mistral-7b-instruct"]
)

# -------------------------------
# Main Evaluation Workflow
# -------------------------------

if uploaded_file:
    input_csv = "input.csv"
    with open(input_csv, "wb") as f:
        f.write(uploaded_file.read())

    output_csv = "results.csv"
    
    # Button to start evaluation
    if st.button("Start Evaluation"):
        st.write("⏳ Running evaluation...")
        
        if model_choice == "gemini-2.5-flash":
            scorer = score_with_gemini
        elif model_choice == "llama-3.1-8b-instant":
            scorer = score_with_groq
        else:
            scorer = score_with_qwen
        
        # Load previously evaluated rows
        full_df = st.session_state.get("full_df", pd.DataFrame())

        # Always evaluate new rows up to row_limit (or all if row_limit=0)
        new_df = evaluate_csv(
            input_csv, 
            output_csv, 
            batch_size, 
            None if row_limit == 0 else row_limit,
            scorer=scorer      
        )

        # Merge with previous results (keep all evaluations)
        if not full_df.empty:
            full_df = pd.concat([full_df, new_df]).drop_duplicates().reset_index(drop=True)
        else:
            full_df = new_df

        # Save all evaluated rows
        st.session_state["full_df"] = full_df

        # Decide what to show
        if row_limit > 0 and len(full_df) > row_limit:
            view_df = full_df.sample(row_limit, random_state=42).reset_index(drop=True)
        else:
            view_df = full_df.copy()
            
        # Save view-specific DataFrame separately for visualization
        st.session_state["view_df"] = view_df

        st.success("✅ Evaluation complete!")
        st.dataframe(view_df)

    # -------------------------------
    # Visualization Tools
    # -------------------------------
    
    # Show stored results if available
    if "view_df" in st.session_state:
        df_result = st.session_state["view_df"]
        print("Rows shown:", len(df_result))
        
        # Evaluation dimensions to visualize
        dims = ["instruction_following", "hallucination", "assumption_control", "coherence_accuracy"]
        # Button: Show correlation heatmap of evaluation dimensions
        if st.button("Show Heatmap"):
            fig, ax = plt.subplots()
            sns.heatmap(df_result[dims].corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title("Correlation Between Evaluation Dimensions")
            st.pyplot(fig)
        
        # Button: Show histogram of total scores
        if st.button("Show Histogram of Total Scores"):
            fig, ax = plt.subplots()
            df_result["total_score"].hist(ax=ax, bins=20,edgecolor="black")
            plt.title("Distribution of Total Scores")
            plt.xlabel("Total Score")
            plt.ylabel("Frequency")
            st.pyplot(fig)
            
        # Button: Show histograms for individual evaluation dimensions    
        if st.button("Show Histogram of dimensions distribution"):
            fig, ax = plt.subplots(2, 2, figsize=(10, 8))
            ax = ax.flatten()
            for i, dim in enumerate(dims):
                df_result[dim].hist(ax=ax[i], bins=10, edgecolor="black")
                ax[i].set_title(f"{dim} distribution")
            plt.tight_layout()
            st.pyplot(fig)
