import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import shap


def get_anomaly_data():
    """Simulates fetching historical data, predictions, and SHAP values."""
    
    timestamps = pd.to_datetime(pd.date_range(start='2025-11-28', periods=100, freq='10min'))
    
    
    data = {
        'timestamp': timestamps,
        'Engine_Temp': np.random.normal(90, 5, 100),
        'Fuel_Level': np.random.uniform(20, 100, 100),
    }
    df = pd.DataFrame(data)
    
    
    df['Anomaly_Score'] = np.random.uniform(0.1, 0.9, 100)
    
    anomaly_indices = [5, 45, 90]
    df.loc[anomaly_indices, 'Anomaly_Score'] = np.random.uniform(0.95, 1.0, len(anomaly_indices))
    df['Prediction'] = np.where(df['Anomaly_Score'] > 0.9, 'Anomaly', 'Normal')
    
    
    anomaly_row = df.iloc[90]
    features = ['Engine_Temp', 'Fuel_Level']
    
   
    shap_values = pd.Series([0.6, -0.1], index=features) 
    
    return df, anomaly_row, shap_values


st.set_page_config(layout="wide", page_title="Fleet Anomaly Monitoring")

st.title("🚛 Fleet Anomaly Monitoring Dashboard")
st.markdown("---")


try:
    data_df, last_anomaly_row, last_shap_values = get_anomaly_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()



col1, col2 = st.columns([1, 2])

with col1:
    st.header("Latest Anomaly Summary")
    
   
    if last_anomaly_row is not None:
        st.metric(
            label="Last Anomaly Time", 
            value=last_anomaly_row['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        )
        st.metric(
            label="Anomaly Score", 
            value=f"{last_anomaly_row['Anomaly_Score']:.4f}",
            delta="Anomaly Detected" if last_anomaly_row['Prediction'] == 'Anomaly' else "Normal"
        )
    
    st.subheader("Recent Anomalies")
    anomaly_table = data_df[data_df['Prediction'] == 'Anomaly'][['timestamp', 'Anomaly_Score']].sort_values(by='timestamp', ascending=False)
    st.dataframe(anomaly_table, use_container_width=True, hide_index=True)



with col2:
    st.header("Feature Time-Series with Anomaly Flags")
    
    selected_feature = st.selectbox(
        "Select Feature to Plot:", 
        options=['Engine_Temp', 'Fuel_Level'],
        key='feature_select'
    )

   
    fig = px.line(data_df, x='timestamp', y=selected_feature, title=f'{selected_feature} Over Time')
    
    
    anomaly_points = data_df[data_df['Prediction'] == 'Anomaly']
    fig.add_trace(go.Scatter(
        x=anomaly_points['timestamp'],
        y=anomaly_points[selected_feature],
        mode='markers',
        name='Anomaly',
        marker=dict(size=10, color='red', symbol='x')
    ))

    fig.update_layout(hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")


st.header("🧠 Anomaly Explanation (XAI - SHAP)")

if last_shap_values is not None:
    st.write(f"Showing feature contributions for the anomaly at **{last_anomaly_row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}**.")

    
    shap_df = last_shap_values.reset_index()
    shap_df.columns = ['Feature', 'SHAP Value (Contribution)']
    shap_df['Color'] = np.where(shap_df['SHAP Value (Contribution)'] > 0, 'High Risk (Positive)', 'Low Risk (Negative)')

    
    colors = {'High Risk (Positive)': 'red', 'Low Risk (Negative)': 'blue'}
    
    shap_fig = px.bar(
        shap_df.sort_values(by='SHAP Value (Contribution)', ascending=False),
        y='Feature',
        x='SHAP Value (Contribution)',
        color='Color',
        orientation='h',
        color_discrete_map=colors,
        title="SHAP Value Contribution to Anomaly"
    )
    
    shap_fig.update_layout(yaxis_title="")
    st.plotly_chart(shap_fig, use_container_width=True)
    
    st.markdown("""
    **Interpretation:**
    * The **SHAP Value** represents how much each feature pushes the model's output (Anomaly Score) away from the average baseline.
    * **Red bars (Positive SHAP Value)** indicate features that are increasing the probability/score of the observation being an **Anomaly**.
    * **Blue bars (Negative SHAP Value)** indicate features that are decreasing the probability/score of the observation being an Anomaly (i.e., making it look more **Normal**).
    """)


if st.button("Refresh Data"):
    st.rerun()