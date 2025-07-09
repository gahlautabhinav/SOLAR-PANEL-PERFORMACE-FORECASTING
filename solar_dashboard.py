import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="☀️ Solar Panel Performance Analytics",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    
    .season-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #ff6b35, #f7931e);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

# Feature ranges and data generation functions
@st.cache_data
def get_feature_ranges():
    return {
        'winter': {
            'irradiance': (300, 700),
            'humidity': (30, 70),
            'wind_speed': (1, 6),
            'ambient_temperature': (5, 20),
            'tilt_angle': (10, 40),
        },
        'summer': {
            'irradiance': (600, 1000),
            'humidity': (50, 90),
            'wind_speed': (3, 8),
            'ambient_temperature': (25, 40),
            'tilt_angle': (0, 20),
        },
        'monsoon': {
            'irradiance': (100, 500),
            'humidity': (70, 100),
            'wind_speed': (5, 10),
            'ambient_temperature': (20, 30),
            'tilt_angle': (20, 30),
        }
    }

def calc_kwh_winter(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.18 * irradiance - 0.03 * humidity + 0.015 * wind_speed + 
            0.08 * ambient_temp - 0.02 * abs(tilt_angle - 30))

def calc_kwh_summer(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.25 * irradiance - 0.05 * humidity + 0.02 * wind_speed + 
            0.1 * ambient_temp - 0.01 * abs(tilt_angle - 10))

def calc_kwh_monsoon(irradiance, humidity, wind_speed, ambient_temp, tilt_angle):
    return (0.1 * irradiance - 0.01 * humidity + 0.03 * wind_speed + 
            0.05 * ambient_temp - 0.01 * abs(tilt_angle - 25))

@st.cache_data
def generate_complete_dataset():
    feature_ranges = get_feature_ranges()
    
    # Months and days configuration
    months_config = {
        'winter': {'November': 30, 'December': 31, 'January': 31, 'February': 28},
        'summer': {'March': 31, 'April': 30, 'May': 31, 'June': 30},
        'monsoon': {'July': 31, 'August': 31, 'September': 30, 'October': 31}
    }
    
    calc_functions = {
        'winter': calc_kwh_winter,
        'summer': calc_kwh_summer,
        'monsoon': calc_kwh_monsoon
    }
    
    all_data = []
    
    for season, months in months_config.items():
        for month, days in months.items():
            for day in range(1, days + 1):
                irr = np.random.uniform(*feature_ranges[season]['irradiance'])
                hum = np.random.uniform(*feature_ranges[season]['humidity'])
                wind = np.random.uniform(*feature_ranges[season]['wind_speed'])
                temp = np.random.uniform(*feature_ranges[season]['ambient_temperature'])
                tilt = np.random.uniform(*feature_ranges[season]['tilt_angle'])
                
                kwh = calc_functions[season](irr, hum, wind, temp, tilt)
                
                all_data.append({
                    'irradiance': round(irr, 2),
                    'humidity': round(hum, 2),
                    'wind_speed': round(wind, 2),
                    'ambient_temperature': round(temp, 2),
                    'tilt_angle': round(tilt, 2),
                    'kwh': round(kwh, 2),
                    'season': season,
                    'month': month,
                    'day': day
                })
    
    return pd.DataFrame(all_data)

# Main app
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1 style="color: white; text-align: center; margin: 0;">
            ☀️ Solar Panel Performance Analytics Dashboard
        </h1>
        <p style="color: white; text-align: center; margin: 0; opacity: 0.9;">
            Advanced Machine Learning Analysis for Solar Energy Optimization
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("### 🔧 Dashboard Controls")
    
    # Generate or load data
    if st.sidebar.button("🔄 Generate New Dataset", key="generate_data"):
        st.cache_data.clear()
    
    df = generate_complete_dataset()
    
    # Sidebar options
    analysis_type = st.sidebar.selectbox(
        "📊 Select Analysis Type",
        ["📈 Data Overview", "🔮 Energy Prediction", "🎯 Season Classification", "📱 Interactive Predictor"]
    )
    
    # Main content based on selection
    if analysis_type == "📈 Data Overview":
        show_data_overview(df)
    elif analysis_type == "🔮 Energy Prediction":
        show_energy_prediction(df)
    elif analysis_type == "🎯 Season Classification":
        show_season_classification(df)
    elif analysis_type == "📱 Interactive Predictor":
        show_interactive_predictor(df)

def show_data_overview(df):
    st.markdown("## 📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🗓️ Total Days</h3>
            <h2>{len(df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3>⚡ Avg kWh</h3>
            <h2>{df['kwh'].mean():.2f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3>🌡️ Avg Temp</h3>
            <h2>{df['ambient_temperature'].mean():.1f}°C</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-container">
            <h3>☀️ Avg Irradiance</h3>
            <h2>{df['irradiance'].mean():.0f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Seasonal energy distribution
        fig_box = px.box(df, x='season', y='kwh', 
                        title='🎭 Energy Output by Season',
                        color='season',
                        color_discrete_map={'summer': '#ff6b35', 'winter': '#74b9ff', 'monsoon': '#00b894'})
        fig_box.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # Monthly trend
        monthly_avg = df.groupby('month')['kwh'].mean().reset_index()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_avg['month'] = pd.Categorical(monthly_avg['month'], categories=month_order, ordered=True)
        monthly_avg = monthly_avg.sort_values('month')
        
        fig_line = px.line(monthly_avg, x='month', y='kwh', 
                          title='📅 Monthly Energy Trend',
                          markers=True)
        fig_line.update_traces(line_color='#ff6b35', line_width=3)
        fig_line.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Feature correlation heatmap
    st.markdown("### 🔗 Feature Correlation Matrix")
    corr_matrix = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']].corr()
    fig_heatmap = px.imshow(corr_matrix, 
                           title='Feature Correlation Heatmap',
                           color_continuous_scale='RdBu',
                           aspect='auto')
    fig_heatmap.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Raw data table
    st.markdown("### 📋 Raw Dataset Sample")
    st.dataframe(df.head(20), use_container_width=True)

def show_energy_prediction(df):
    st.markdown("## 🔮 Energy Output Prediction Model")
    
    # Prepare data
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
    y = df['kwh']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Model performance
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>📊 R² Score</h3>
            <h2>{r2:.4f}</h2>
            <p>Model Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>📉 MSE</h3>
            <h2>{mse:.4f}</h2>
            <p>Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="prediction-box">
            <h3>🎯 RMSE</h3>
            <h2>{np.sqrt(mse):.4f}</h2>
            <p>Root Mean Squared Error</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction vs Actual plot
    col1, col2 = st.columns(2)
    
    with col1:
        fig_scatter = px.scatter(x=y_test, y=y_pred, 
                               title='🎯 Actual vs Predicted Energy Output',
                               labels={'x': 'Actual kWh', 'y': 'Predicted kWh'})
        
        # Add diagonal line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                       mode='lines', name='Perfect Prediction',
                                       line=dict(color='red', dash='dash')))
        
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Residuals plot
        residuals = y_test - y_pred
        fig_residuals = px.scatter(x=y_pred, y=residuals,
                                 title='📊 Residuals Plot',
                                 labels={'x': 'Predicted kWh', 'y': 'Residuals'})
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residuals.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white')
        )
        st.plotly_chart(fig_residuals, use_container_width=True)
    
    # Feature importance
    st.markdown("### 🎯 Feature Importance")
    feature_names = ['Irradiance', 'Humidity', 'Wind Speed', 'Temperature', 'Tilt Angle']
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_,
        'Abs_Coefficient': np.abs(model.coef_)
    }).sort_values('Abs_Coefficient', ascending=True)
    
    fig_importance = px.bar(importance_df, x='Abs_Coefficient', y='Feature',
                           title='Feature Importance (Absolute Coefficients)',
                           orientation='h')
    fig_importance.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_importance, use_container_width=True)

def show_season_classification(df):
    st.markdown("## 🎯 Season Classification Model")
    
    # Prepare data
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle', 'kwh']]
    y = df['season']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Model performance
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🎯 Classification Accuracy</h3>
        <h2>{accuracy:.4f}</h2>
        <p>Model correctly predicts seasons {accuracy*100:.2f}% of the time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    fig_cm = px.imshow(cm, 
                       x=le.classes_, 
                       y=le.classes_,
                       title='🔍 Confusion Matrix',
                       color_continuous_scale='Blues',
                       aspect='auto')
    fig_cm.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification report
    st.markdown("### 📊 Detailed Classification Report")
    report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)

def show_interactive_predictor(df):
    st.markdown("## 📱 Interactive Energy Predictor")
    
    # Train the model
    X = df[['irradiance', 'humidity', 'wind_speed', 'ambient_temperature', 'tilt_angle']]
    y = df['kwh']
    model = LinearRegression()
    model.fit(X, y)
    
    # Interactive inputs
    st.markdown("### 🎛️ Adjust Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        irradiance = st.slider("☀️ Solar Irradiance", 100, 1000, 500, help="Solar irradiance in W/m²")
        humidity = st.slider("💧 Humidity (%)", 20, 100, 60, help="Relative humidity percentage")
        wind_speed = st.slider("💨 Wind Speed (m/s)", 0, 15, 5, help="Wind speed in meters per second")
    
    with col2:
        temperature = st.slider("🌡️ Temperature (°C)", 0, 45, 25, help="Ambient temperature in Celsius")
        tilt_angle = st.slider("📐 Tilt Angle (°)", 0, 50, 25, help="Solar panel tilt angle in degrees")
    
    # Make prediction
    prediction = model.predict([[irradiance, humidity, wind_speed, temperature, tilt_angle]])[0]
    
    # Display prediction
    st.markdown(f"""
    <div class="prediction-box">
        <h3>⚡ Predicted Energy Output</h3>
        <h1 style="font-size: 3em;">{prediction:.2f} kWh</h1>
        <p>Based on the input parameters</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show parameter impact
    st.markdown("### 📊 Parameter Impact Analysis")
    
    # Calculate how each parameter affects the prediction
    base_params = [irradiance, humidity, wind_speed, temperature, tilt_angle]
    impacts = []
    
    for i, param_name in enumerate(['Irradiance', 'Humidity', 'Wind Speed', 'Temperature', 'Tilt Angle']):
        modified_params = base_params.copy()
        modified_params[i] *= 1.1  # Increase by 10%
        new_prediction = model.predict([modified_params])[0]
        impact = ((new_prediction - prediction) / prediction) * 100
        impacts.append({'Parameter': param_name, 'Impact (%)': impact})
    
    impact_df = pd.DataFrame(impacts)
    fig_impact = px.bar(impact_df, x='Parameter', y='Impact (%)',
                       title='📈 10% Parameter Increase Impact on Energy Output',
                       color='Impact (%)',
                       color_continuous_scale='RdBu')
    fig_impact.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig_impact, use_container_width=True)
    
    # Recommendations
    st.markdown("### 💡 Optimization Recommendations")
    
    recommendations = []
    if irradiance < 600:
        recommendations.append("☀️ Consider relocating panels to area with higher solar irradiance")
    if tilt_angle < 20 or tilt_angle > 35:
        recommendations.append("📐 Optimize tilt angle to 25-30° for better performance")
    if humidity > 80:
        recommendations.append("💧 High humidity may reduce efficiency - ensure proper ventilation")
    if wind_speed < 2:
        recommendations.append("💨 Low wind speed may cause overheating - consider cooling solutions")
    
    if recommendations:
        for rec in recommendations:
            st.info(rec)
    else:
        st.success("🎉 Your parameters are well-optimized for solar energy generation!")

if __name__ == "__main__":
    main()
