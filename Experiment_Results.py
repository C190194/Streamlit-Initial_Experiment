import streamlit as st
import pandas as pd
import numpy as np

st.title("Experiment Results using Baseline Methods \
        and Team JKU's Methods") 

def color_overall_row(s):
        if s["Machine Type"] == "Overall":
                color = "#b1e5f2"
        else:
                color = "transparent"
        return [f'background-color: {color}']*len(s)
        
        

st.subheader("Results for Full Development Dataset \
        Using Autoencoder") 

ae_df = pd.read_csv ('Baseline_AE_Results.csv')
st.table(ae_df.style.format({"Source Hmean": "{:.2f}",
                             "Target Hmean": "{:.2f}",
                             "AUC Hmean": "{:.2f}",
                             "pAUC Hmean": "{:.2f}"}
                        ).apply(color_overall_row, axis=1))

st.subheader("Results for Full Development Dataset \
        MobileNetV2")

mn_df = pd.read_csv ('Baseline_MobileNetV2_Results.csv')
st.table(mn_df.style.format({"Source Hmean": "{:.2f}",
                             "Target Hmean": "{:.2f}",
                             "AUC Hmean": "{:.2f}",
                             "pAUC Hmean": "{:.2f}"}
                        ).apply(color_overall_row, axis=1))

st.subheader("Results for Fan's Development Dataset \
        Using Auxiliary Classification and Density Estimation")

jku_df = pd.read_csv ('Primus_Fan_Results.csv')
st.table(jku_df.style.format({"Source Hmean": "{:.1f}",
                             "Target Hmean": "{:.1f}",
                             "JKU Source Hmean": "{:.1f}",
                             "JKU Target Hmean": "{:.1f}"}
                        ))
st.write("MADE - Masked Autoencoder for Distribution Estimation")
st.write("MAF - Masked Autoregressive Flows")
st.write("AC - Auxiliary Classification")
st.write("FT - Fine Tuninng")  