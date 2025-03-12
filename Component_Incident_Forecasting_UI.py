import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import warnings
import io
import base64
import pickle
from prophet.serialize import model_to_json, model_from_json

warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Component Incident Forecasting Tool",
    page_icon="📊",
    layout="wide"
)

# App title and description
st.title("📊 Component Incident Forecasting Tool")
st.markdown("""
This application predicts future incident trends for selected components using multiple forecasting models (Holt-Winters, ARIMA, and Prophet).
The models are combined to create an optimized forecast with weights based on validation performance.
""")

# Sidebar for data upload and component selection
st.sidebar.header("Data Input")

# Function to load and preprocess data
@st.cache_data


# ===== Helper Functions =====
def check_stationarity(data):
    result = adfuller(data.dropna())
    return result[1] < 0.05

def get_optimal_arima_orders(train_data):
    acf_values = acf(train_data, nlags=min(20, len(train_data) // 2))
    pacf_values = pacf(train_data, nlags=min(20, len(train_data) // 2))
    p = np.where(np.abs(pacf_values) > 0.2)[0][-1] if len(np.where(np.abs(pacf_values) > 0.2)[0]) > 0 else 1
    q = np.where(np.abs(acf_values) > 0.2)[0][-1] if len(np.where(np.abs(acf_values) > 0.2)[0]) > 0 else 1
    return min(p, 5), min(q, 5)

def check_overfitting_metrics(train_data, val_data, test_data, predictions_train, predictions_val, predictions_test):
    train_mae = mean_absolute_error(train_data, predictions_train)
    val_mae = mean_absolute_error(val_data, predictions_val)
    test_mae = mean_absolute_error(test_data, predictions_test)
    val_train_ratio = val_mae / train_mae if train_mae > 0 else float('inf')
    test_train_ratio = test_mae / train_mae if train_mae > 0 else float('inf')
    if val_train_ratio > 2 or test_train_ratio > 2:
        status = "High risk of overfitting"
    elif val_train_ratio > 1.5 or test_train_ratio > 1.5:
        status = "Moderate risk of overfitting"
    else:
        status = "Low risk of overfitting"
    return {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'test_mae': test_mae,
        'val_train_ratio': val_train_ratio,
        'test_train_ratio': test_train_ratio,
        'status': status
    }

def calculate_accuracy_metrics(actual, predicted):
    valid_mask = ~np.isnan(actual) & ~np.isnan(predicted)
    actual = actual[valid_mask]
    predicted = predicted[valid_mask]
    if len(actual) == 0 or len(predicted) == 0:
        return {k: np.nan for k in ['MSE', 'RMSE', 'MAE', 'MAPE', 'SMAPE']}
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    mape = np.mean(np.abs((actual - predicted) / np.where(actual != 0, actual, np.inf))) * 100
    denom = (np.abs(actual) + np.abs(predicted))
    denom[denom == 0] = 1
    smape = np.mean(2 * np.abs(actual - predicted) / denom) * 100
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'SMAPE': smape}

def calculate_weights(hw_val_mae, arima_val_mae):
    total_error = hw_val_mae + arima_val_mae
    if total_error == 0:
        return 0.5, 0.5
    hw_weight = 1 - (hw_val_mae / total_error)
    arima_weight = 1 - (arima_val_mae / total_error)
    total_weight = hw_weight + arima_weight
    return hw_weight / total_weight, arima_weight / total_weight

def train_holtwinters_model(train_data, seasonal_periods=[3, 6, 12]):
    hw_best_mae = float('inf')
    hw_best_model = None
    for sp in seasonal_periods:
        if sp < len(train_data) - 1:
            for trend in ['add', 'mul']:
                for seasonal in ['add', 'mul']:
                    try:
                        model = ExponentialSmoothing(train_data, trend=trend, seasonal=seasonal,
                                                    seasonal_periods=sp, damped=True).fit(optimized=True)
                        forecast = model.forecast(len(train_data))
                        mae = mean_absolute_error(train_data, forecast)
                        if mae < hw_best_mae:
                            hw_best_mae = mae
                            hw_best_model = model
                    except Exception as e:
                        continue
    return hw_best_model, hw_best_mae

def train_arima_model(train_data, p_optimal, q_optimal):
    arima_best_mae = float('inf')
    arima_best_model = None
    for i in range(p_optimal + 1):
        for d in range(2):
            for j in range(q_optimal + 1):
                try:
                    model = ARIMA(train_data, order=(i, d, j)).fit()
                    forecast = model.forecast(steps=len(train_data))
                    mae = mean_absolute_error(train_data, forecast)
                    if mae < arima_best_mae:
                        arima_best_mae = mae
                        arima_best_model = model
                except Exception as e:
                    continue
    return arima_best_model, arima_best_mae

def train_prophet_model(train_data):
    prophet_best_mae = float('inf')
    prophet_best_model = None
    changepoint_priors = [0.05, 0.1, 0.2]
    for prior in changepoint_priors:
        for yearly_seasonality in [True, False]:
            try:
                model = Prophet(yearly_seasonality=yearly_seasonality, daily_seasonality=False,
                               changepoint_prior_scale=prior, seasonality_prior_scale=10)
                model.add_seasonality(name='quarterly', period=90, fourier_order=5)
                model.fit(train_data)
                forecast = model.predict(train_data)
                mae = mean_absolute_error(train_data['y'].values, forecast['yhat'].values)
                if mae < prophet_best_mae:
                    prophet_best_mae = mae
                    prophet_best_model = model
            except Exception as e:
                continue
    return prophet_best_model, prophet_best_mae


def load_and_preprocess_data(file):
    """
    Load and preprocess the incident data from uploaded CSV file
    Parameters:
        file: Uploaded CSV file
    Returns:
        pd.DataFrame: Preprocessed monthly trends data
        list: Components in the data
    """
    # Load data
    data = pd.read_csv(file)
    data['Creation Month'] = pd.to_datetime(data['Creation Month'], format='%Y-%m')

    # Group and aggregate data
    monthly_trends = data.groupby(['Creation Month', 'Component'])['Incident'].count().reset_index()
    component_list = monthly_trends['Component'].unique().tolist()

    return monthly_trends, component_list

# Data upload
uploaded_file = st.sidebar.file_uploader("Upload incident data CSV", type=["csv"])
st.sidebar.header("Or Upload Saved Model")
uploaded_model = st.sidebar.file_uploader("Upload saved model file", type=["pkl", "json"])

if uploaded_model is not None:
    st.sidebar.header("Model Settings")
    forecast_months = st.sidebar.slider("Forecast Months", 3, 12, 6, key="forecast_months_advanced")
    
    if st.sidebar.button("Generate Forecast from Saved Model"):
        with st.spinner("Loading model and generating forecast..."):
            model_file_ext = uploaded_model.name.split('.')[-1]
            
            if model_file_ext == "json":
                model_bytes = uploaded_model.read()
                prophet_model = model_from_json(model_bytes.decode())
                future_dates = pd.date_range(
                    start=pd.Timestamp.now(), 
                    periods=forecast_months, 
                    freq='MS'
                )
                future = pd.DataFrame({'ds': future_dates})
                forecast = prophet_model.predict(future)
                
                st.subheader("Forecast from Uploaded Prophet Model")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(future_dates, forecast['yhat'], 
                        label='Prophet Forecast', marker='o', markersize=4, color='red')
                ax.fill_between(future_dates, 
                                forecast['yhat_lower'], 
                                forecast['yhat_upper'], 
                                color='red', alpha=0.2, label='Confidence Interval')
                ax.set_title('Future Forecast from Uploaded Prophet Model', fontsize=16, pad=20)
                ax.set_xlabel('Date', fontsize=12)
                ax.set_ylabel('Predicted Incidents', fontsize=12)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                
                forecast_df = pd.DataFrame({
                    'Date': forecast['ds'],
                    'Predicted': forecast['yhat'],
                    'Lower Bound': forecast['yhat_lower'],
                    'Upper Bound': forecast['yhat_upper']
                })
                st.dataframe(forecast_df)
                
            elif model_file_ext == "pkl":
                model = pickle.load(uploaded_model)
                future_dates = pd.date_range(
                    start=pd.Timestamp.now(), 
                    periods=forecast_months, 
                    freq='MS'
                )
                
                if isinstance(model, dict) and 'hw_model' in model:
                    hw_model = model['hw_model']
                    arima_model = model['arima_model']
                    hw_weight = model['hw_weight']
                    arima_weight = model['arima_weight']
                    
                    hw_forecast = hw_model.forecast(forecast_months)
                    arima_forecast = arima_model.forecast(steps=forecast_months)
                    combined_forecast = hw_forecast * hw_weight + arima_forecast * arima_weight
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(future_dates, hw_forecast, 
                            label='Holt-Winters Forecast', linestyle='--', marker='o', markersize=4, color='orange')
                    ax.plot(future_dates, arima_forecast, 
                            label='ARIMA Forecast', linestyle='--', marker='o', markersize=4, color='green')
                    ax.plot(future_dates, combined_forecast, 
                            label='Combined Forecast', linestyle='-', marker='o', markersize=4, color='purple')
                    ax.set_title('Future Forecast from Uploaded Combined Model', fontsize=16, pad=20)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Predicted Incidents', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Holt-Winters': hw_forecast.values,
                        'ARIMA': arima_forecast.values,
                        'Combined': combined_forecast.values
                    })
                    st.dataframe(forecast_df)
                    
                elif str(type(model)).find('ExponentialSmoothing') >= 0:
                    forecast = model.forecast(forecast_months)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(future_dates, forecast, 
                            label='Holt-Winters Forecast', marker='o', markersize=4, color='orange')
                    ax.set_title('Future Forecast from Uploaded Holt-Winters Model', fontsize=16, pad=20)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Predicted Incidents', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'Holt-Winters Forecast': forecast.values
                    })
                    st.dataframe(forecast_df)
                    
                elif str(type(model)).find('ARIMA') >= 0:
                    forecast = model.forecast(steps=forecast_months)
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(future_dates, forecast, 
                            label='ARIMA Forecast', marker='o', markersize=4, color='green')
                    ax.set_title('Future Forecast from Uploaded ARIMA Model', fontsize=16, pad=20)
                    ax.set_xlabel('Date', fontsize=12)
                    ax.set_ylabel('Predicted Incidents', fontsize=12)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    forecast_df = pd.DataFrame({
                        'Date': future_dates,
                        'ARIMA Forecast': forecast.values
                    })
                    st.dataframe(forecast_df)
                    
                else:
                    st.error("Unknown model type. Please upload a valid model file.")
            
            else:
                st.error("Unsupported file format. Please upload a .pkl or .json file.")
                
            csv = forecast_df.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="forecast_from_uploaded_model.csv">Download Forecast CSV</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

            

if uploaded_file is not None:
    monthly_trends, component_list = load_and_preprocess_data(uploaded_file)
    
    st.sidebar.header("Component Selection")
    selected_component = st.sidebar.selectbox(
        "Select a component to analyze",
        options=component_list
    )
    
    with st.sidebar.expander("Advanced Settings"):
        forecast_months = st.slider("Forecast Months", 3, 12, 6, key="forecast_months_model")
        train_test_split = st.slider("Training Data %", 50, 90, 80)
        use_prophet = st.checkbox("Include Prophet Model", value=True)
    
    if st.sidebar.button("Generate Forecast"):
        with st.spinner(f"Analyzing component: {selected_component}..."):
            st.subheader(f"Analysis for Component: {selected_component}")
            
            def prepare_component_data(monthly_trends, component):
                component_data = monthly_trends[monthly_trends['Component'] == component]
                date_range = pd.date_range(component_data['Creation Month'].min(),
                                        component_data['Creation Month'].max(), 
                                        freq='MS')
                component_data = component_data.set_index('Creation Month').reindex(date_range, fill_value=0)
                component_data.index.name = 'Creation Month'

                q1 = component_data['Incident'].quantile(0.10)
                q3 = component_data['Incident'].quantile(0.90)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                component_data = component_data[(component_data['Incident'] >= lower_bound) &
                                            (component_data['Incident'] <= upper_bound)]

                component_data['Smoothed_Incident'] = component_data['Incident'].rolling(window=3, center=True).mean()
                component_data['Incident'].fillna(component_data['Smoothed_Incident'], inplace=True)

                scaler = MinMaxScaler()
                component_data['Scaled Incident'] = scaler.fit_transform(component_data[['Incident']])

                return component_data, scaler

            component_data, scaler = prepare_component_data(monthly_trends, selected_component)
            time_series_data = component_data['Scaled Incident']
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Historical Incident Data")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(component_data.index, component_data['Incident'], marker='o', linestyle='-')
                ax.set_title(f"Historical Incidents: {selected_component}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Number of Incidents")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Scaled Incident Data")
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(time_series_data.index, time_series_data, marker='o', linestyle='-', color='green')
                ax.set_title(f"Scaled Incidents: {selected_component}")
                ax.set_xlabel("Date")
                ax.set_ylabel("Scaled Incidents")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)

            stationarity_result = check_stationarity(time_series_data)
            if not stationarity_result:
                st.info("Time series is not stationary. Differencing applied.")
                time_series_data = time_series_data.diff().dropna()
            else:
                st.info("Time series is stationary. No differencing needed.")

            train_size = int(len(time_series_data) * train_test_split / 100)
            train_data = time_series_data.iloc[:train_size]
            test_data = time_series_data.iloc[train_size:]
            val_size = len(train_data) // 3
            train_subset = train_data.iloc[:-val_size]
            val_data = train_data.iloc[-val_size:]
            
            st.info(f"Data split: Training ({train_size} points), Testing ({len(test_data)} points)")

            progress_bar = st.progress(0)
            progress_text = st.empty()
            
            progress_text.text("Training Holt-Winters model...")
            progress_bar.progress(10)
            hw_best_model, _ = train_holtwinters_model(train_data)
            hw_forecast = hw_best_model.forecast(len(test_data))
            hw_future_forecast = hw_best_model.forecast(forecast_months)
            hw_train_pred = hw_best_model.fittedvalues[-len(train_subset):]
            hw_val_pred = hw_best_model.forecast(len(val_data))
            
            progress_bar.progress(30)
            
            progress_text.text("Training ARIMA model...")
            p_optimal, q_optimal = get_optimal_arima_orders(train_data)
            arima_best_model, _ = train_arima_model(train_data, p_optimal, q_optimal)
            arima_forecast = arima_best_model.forecast(steps=len(test_data))
            arima_future_forecast = arima_best_model.forecast(steps=forecast_months)
            arima_train_pred = arima_best_model.fittedvalues[-len(train_subset):]
            arima_val_pred = arima_best_model.forecast(steps=len(val_data))
            
            progress_bar.progress(60)

            prophet_best_model = None
            prophet_forecast = None
            prophet_future_forecast = None
            prophet_train_pred = None
            prophet_val_pred = None
            
            if use_prophet:
                progress_text.text("Training Prophet model...")
                prophet_data = train_data.reset_index()
                prophet_data.columns = ['ds', 'y']
                
                try:
                    prophet_best_model, _ = train_prophet_model(prophet_data)
                    
                    if prophet_best_model is not None:
                        future = pd.DataFrame({'ds': test_data.index})
                        prophet_forecast_result = prophet_best_model.predict(future)
                        prophet_forecast = prophet_forecast_result['yhat']
                        
                        if len(prophet_forecast) < len(test_data):
                            st.warning("Prophet forecast is shorter than test data. Padding with NaN values.")
                            prophet_forecast = np.pad(prophet_forecast, (0, len(test_data) - len(prophet_forecast)), mode='constant', constant_values=np.nan)
                        elif len(prophet_forecast) > len(test_data):
                            st.warning("Prophet forecast is longer than test data. Truncating forecast.")
                            prophet_forecast = prophet_forecast[:len(test_data)]
                        
                        future_dates = pd.date_range(
                            start=time_series_data.index[-1] + pd.DateOffset(months=1), 
                            periods=forecast_months, 
                            freq='MS'
                        )
                        future_prophet = pd.DataFrame({'ds': future_dates})
                        prophet_future_result = prophet_best_model.predict(future_prophet)
                        prophet_future_forecast = prophet_future_result['yhat']
                        
                        train_prophet = pd.DataFrame({'ds': train_subset.index})
                        val_prophet = pd.DataFrame({'ds': val_data.index})
                        prophet_train_result = prophet_best_model.predict(train_prophet)
                        prophet_val_result = prophet_best_model.predict(val_prophet)
                        prophet_train_pred = prophet_train_result['yhat'].values
                        prophet_val_pred = prophet_val_result['yhat'].values
                except Exception as e:
                    st.warning(f"Prophet model training failed: {str(e)}")
                    use_prophet = False
            
            progress_bar.progress(80)
            progress_text.text("Calculating metrics and preparing visualizations...")

            hw_metrics = check_overfitting_metrics(
                train_subset[-len(hw_train_pred):], 
                val_data, 
                test_data,
                hw_train_pred,
                hw_val_pred,
                hw_forecast
            )

            arima_metrics = check_overfitting_metrics(
                train_subset[-len(arima_train_pred):],
                val_data,
                test_data,
                arima_train_pred,
                arima_val_pred,
                arima_forecast
            )

            if use_prophet and prophet_best_model is not None:
                prophet_metrics = check_overfitting_metrics(
                    train_subset,
                    val_data,
                    test_data,
                    prophet_train_pred,
                    prophet_val_pred,
                    prophet_forecast
                )

            hw_weight, arima_weight = calculate_weights(hw_metrics['val_mae'], arima_metrics['val_mae'])
            combined_forecast = hw_forecast * hw_weight + arima_forecast * arima_weight
            combined_future_forecast = hw_future_forecast * hw_weight + arima_future_forecast * arima_weight
            combined_train_pred = hw_train_pred * hw_weight + arima_train_pred * arima_weight
            combined_val_pred = hw_val_pred * hw_weight + arima_val_pred * arima_weight

            combined_metrics = check_overfitting_metrics(
                train_subset[-len(combined_train_pred):],
                val_data,
                test_data,
                combined_train_pred,
                combined_val_pred,
                combined_forecast
            )

            hw_accuracy = calculate_accuracy_metrics(test_data, hw_forecast)
            arima_accuracy = calculate_accuracy_metrics(test_data, arima_forecast)
            combined_accuracy = calculate_accuracy_metrics(test_data, combined_forecast)
            
            if use_prophet and prophet_best_model is not None:
                                prophet_accuracy = calculate_accuracy_metrics(test_data, prophet_forecast)

            progress_bar.progress(100)
            progress_text.empty()
            
            # ===== Visualization =====
            st.subheader("Forecast Results")
            
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(time_series_data.index, time_series_data, 
                    label='Historical Data', marker='o', markersize=4, color='blue', alpha=0.6)
            ax.plot(test_data.index, hw_forecast, 
                    label='Holt-Winters Forecast', linestyle='--', marker='o', markersize=4, color='orange')
            ax.plot(test_data.index, arima_forecast, 
                    label='ARIMA Forecast', linestyle='--', marker='o', markersize=4, color='green')
            ax.plot(test_data.index, combined_forecast, 
                    label='Combined Forecast', linestyle='--', marker='o', markersize=4, color='purple')
            
            if use_prophet and prophet_best_model is not None:
                ax.plot(test_data.index, prophet_forecast, 
                        label='Prophet Forecast', linestyle='--', marker='o', markersize=4, color='red')
            
            ax.axvline(x=train_data.index[-1], color='black', linestyle='--', label='Train/Test Split')
            ax.set_title(f'Forecast Comparison for Component: {selected_component}', fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Scaled Number of Incidents', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Create future forecast visualization with actual values
            future_dates = pd.date_range(
                start=time_series_data.index[-1] + pd.DateOffset(months=1), 
                periods=forecast_months, 
                freq='MS'
            )
            
            # Inverse transform forecasts for actual values
            def inverse_transform_forecast(scaled_forecast, scaler):
                """Convert scaled forecast back to actual incident counts"""
                return scaler.inverse_transform(
                    scaled_forecast.values.reshape(-1, 1)
            ).flatten()

            # Create future results dataframe with both scaled and actual values
            future_results = pd.DataFrame({
                'Date': future_dates,
                # Holt-Winters
                'Holt-Winters (Scaled)': hw_future_forecast.values,
                'Holt-Winters (Actual)': inverse_transform_forecast(hw_future_forecast, scaler),
                # ARIMA
                'ARIMA (Scaled)': arima_future_forecast.values,
                'ARIMA (Actual)': inverse_transform_forecast(arima_future_forecast, scaler),
                # Combined
                'Combined (Scaled)': combined_future_forecast.values,
                'Combined (Actual)': inverse_transform_forecast(combined_future_forecast, scaler)
            })

            if use_prophet and prophet_best_model is not None and prophet_future_forecast is not None:
                # Prophet doesn't use scaling, so actual = scaled
                future_results['Prophet (Actual)'] = prophet_future_forecast.values
                future_results['Prophet (Scaled)'] = prophet_future_forecast.values

            # Plotting future forecasts with actual values
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot historical actual values
            ax.plot(component_data.index, component_data['Incident'], 
                    label='Historical Actual', marker='o', markersize=4, color='blue', alpha=0.6)
            
            # Plot future predictions (actual scale)
            ax.plot(future_dates, future_results['Holt-Winters (Actual)'], 
                    label='Holt-Winters Forecast', linestyle='--', marker='o', markersize=4, color='orange')
            ax.plot(future_dates, future_results['ARIMA (Actual)'], 
                    label='ARIMA Forecast', linestyle='--', marker='o', markersize=4, color='green')
            ax.plot(future_dates, future_results['Combined (Actual)'], 
                    label='Combined Forecast', linestyle='--', marker='o', markersize=4, color='purple')
            
            if use_prophet and prophet_best_model is not None:
                ax.plot(future_dates, future_results['Prophet (Actual)'], 
                        label='Prophet Forecast', linestyle='--', marker='o', markersize=4, color='red')
            
            ax.axvline(x=time_series_data.index[-1], color='black', linestyle='--', label='Current Date')
            ax.set_title(f'Future Forecast (Actual Values) for {selected_component} ({forecast_months} months)', fontsize=16, pad=20)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Number of Incidents', fontsize=12)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

            # ===== Metrics Display =====
            st.subheader("Model Performance Metrics")
            
            # Display model weights
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Model Weights in Combined Forecast")
                weight_fig, weight_ax = plt.subplots(figsize=(6, 4))
                weights = [hw_weight, arima_weight]
                labels = ['Holt-Winters', 'ARIMA']
                weight_ax.pie(weights, labels=labels, autopct='%1.2f%%', startangle=90, colors=['orange', 'green'])
                weight_ax.set_title('Model Contribution to Combined Forecast')
                st.pyplot(weight_fig)
            
            with col2:
                st.markdown("#### Combined Model Accuracy")
                metrics = combined_accuracy
                metric_names = ['RMSE', 'MAE']
                metric_values = [metrics['RMSE'], metrics['MAE']]
                
                acc_fig, acc_ax = plt.subplots(figsize=(6, 4))
                acc_ax.bar(metric_names, metric_values, color='purple')
                acc_ax.set_title('Combined Model Error Metrics')
                acc_ax.set_ylabel('Error Value')
                acc_ax.grid(axis='y', alpha=0.3)
                st.pyplot(acc_fig)
            
            # Create tabs for detailed metrics
            tabs = st.tabs(["Combined Model", "Holt-Winters", "ARIMA", "Prophet" if use_prophet else ""])
            
            def display_model_metrics(tab, name, accuracy, metrics, color):
                with tab:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"#### {name} Model Accuracy Metrics")
                        st.markdown(f"""
                        - **MSE:** {accuracy['MSE']:.4f}
                        - **RMSE:** {accuracy['RMSE']:.4f}
                        - **MAE:** {accuracy['MAE']:.4f}
                        - **MAPE:** {f"{accuracy['MAPE']:.4f}%" if not np.isnan(accuracy['MAPE']) else 'N/A'}
                        - **SMAPE:** {f"{accuracy['SMAPE']:.4f}%" if not np.isnan(accuracy['SMAPE']) else 'N/A'}
                        """)
                    
                    with col2:
                        st.markdown(f"#### {name} Overfitting Analysis")
                        st.markdown(f"""
                        - **Train MAE:** {metrics['train_mae']:.4f}
                        - **Validation MAE:** {metrics['val_mae']:.4f}
                        - **Test MAE:** {metrics['test_mae']:.4f}
                        - **Validation/Train Ratio:** {metrics['val_train_ratio']:.4f}
                        - **Test/Train Ratio:** {metrics['test_train_ratio']:.4f}
                        - **Status:** {metrics['status']}
                        """)
                    
                    # Visual representation of overfitting status
                    st.markdown("#### Overfitting Risk Assessment")
                    if "Low" in metrics['status']:
                        st.success(metrics['status'])
                    elif "Moderate" in metrics['status']:
                        st.warning(metrics['status'])
                    else:
                        st.error(metrics['status'])
            
            # Display metrics for each model
            display_model_metrics(tabs[0], "Combined", combined_accuracy, combined_metrics, 'purple')
            display_model_metrics(tabs[1], "Holt-Winters", hw_accuracy, hw_metrics, 'orange')
            display_model_metrics(tabs[2], "ARIMA", arima_accuracy, arima_metrics, 'green')
            
            if use_prophet and prophet_best_model is not None:
                display_model_metrics(tabs[3], "Prophet", prophet_accuracy, prophet_metrics, 'red')

            # ===== Results Summary =====
            st.subheader("Forecast Summary")
            
            # Create a summary table
            summary_data = {
                'Model': ['Holt-Winters', 'ARIMA', 'Combined'],
                'RMSE': [hw_accuracy['RMSE'], arima_accuracy['RMSE'], combined_accuracy['RMSE']],
                'MAE': [hw_accuracy['MAE'], arima_accuracy['MAE'], combined_accuracy['MAE']],
                'Weight': [hw_weight, arima_weight, 1.0],
                'Overfitting Risk': [hw_metrics['status'], arima_metrics['status'], combined_metrics['status']]
            }
            
            if use_prophet and prophet_best_model is not None:
                summary_data['Model'].append('Prophet')
                summary_data['RMSE'].append(prophet_accuracy['RMSE'])
                summary_data['MAE'].append(prophet_accuracy['MAE'])
                summary_data['Weight'].append(0.0)
                summary_data['Overfitting Risk'].append(prophet_metrics['status'])
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            

            # ===== Export Results =====
            st.subheader("Export Results")
            
            # Create downloadable CSV with both scaled and actual values
            csv = future_results.to_csv(index=False)
            b64_csv = base64.b64encode(csv.encode()).decode()
            href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="forecast_{selected_component}.csv">Download Forecast CSV</a>'
            st.markdown(href_csv, unsafe_allow_html=True)

            
            # Create future results dataframe
            future_results = pd.DataFrame({
                'Date': future_dates,
                'Holt-Winters': hw_future_forecast.values,
                'ARIMA': arima_future_forecast.values,
                'Combined': combined_future_forecast.values
            })
            
            if use_prophet and prophet_best_model is not None and prophet_future_forecast is not None:
                future_results['Prophet'] = prophet_future_forecast.values
            
            # Create a buffer for model saving
            model_buffer = io.BytesIO()
            
            # Determine and save the best model
            best_model = None
            best_model_name = "Combined"  # Default to combined model
            model_extension = ".pkl"
            
            # Compare model performance
            model_performance = {
                'Holt-Winters': hw_accuracy['RMSE'],
                'ARIMA': arima_accuracy['RMSE'],
                'Combined': combined_accuracy['RMSE']
            }
            
            if use_prophet and prophet_best_model is not None:
                model_performance['Prophet'] = prophet_accuracy['RMSE']
            
            # Find model with lowest RMSE
            best_model_name = min(model_performance, key=model_performance.get)
            
            # Save the appropriate model
            if best_model_name == 'Holt-Winters':
                pickle.dump(hw_best_model, model_buffer)
                model_extension = "_hw.pkl"
            elif best_model_name == 'ARIMA':
                pickle.dump(arima_best_model, model_buffer)
                model_extension = "_arima.pkl"
            elif best_model_name == 'Prophet':
                # Prophet requires special serialization
                model_buffer = io.BytesIO()
                model_buffer.write(model_to_json(prophet_best_model).encode())
                model_extension = "_prophet.json"
            else:  # Combined model (save all models and weights)
                combined_models = {
                    'hw_model': hw_best_model,
                    'arima_model': arima_best_model,
                    'hw_weight': hw_weight,
                    'arima_weight': arima_weight
                }
                if use_prophet:
                    combined_models['prophet_model'] = prophet_best_model
                pickle.dump(combined_models, model_buffer)
                model_extension = "_combined.pkl"
            
            # Create downloadable files
           # st.markdown("#### Download Forecast Data")
            #csv = future_results.to_csv(index=False)
           # b64_csv = base64.b64encode(csv.encode()).decode()
           # href_csv = f'<a href="data:file/csv;base64,{b64_csv}" download="forecast_{selected_component}.csv">Download Forecast CSV</a>'
            #st.markdown(href_csv, unsafe_allow_html=True)
            
            st.markdown("#### Download Best Model")
            model_buffer.seek(0)
            b64_model = base64.b64encode(model_buffer.read()).decode()
            href_model = f'<a href="data:application/octet-stream;base64,{b64_model}" download="best_model_{selected_component}{model_extension}">Download {best_model_name} Model</a>'
            st.markdown(href_model, unsafe_allow_html=True)
            
            st.markdown("""
            **Note on Model Download:**
            - Saved models contain all parameters needed for future predictions
            - To reload:
                - For .pkl files: Use `pickle.load(file)`
                - For .json files: Use `Prophet.load(file)`
            - Maintain same library versions for consistent results
            """)

            # ===== Model Interpretation =====
            st.subheader("Model Interpretation")
            model_names = ['Holt-Winters', 'ARIMA', 'Combined']
            model_errors = [hw_accuracy['RMSE'], arima_accuracy['RMSE'], combined_accuracy['RMSE']]
            
            if use_prophet and prophet_best_model is not None:
                model_names.append('Prophet')
                model_errors.append(prophet_accuracy['RMSE'])
            
            best_model_idx = np.argmin(model_errors)
            best_model_name = model_names[best_model_idx]
            best_model_error = model_errors[best_model_idx]

            st.markdown(f"""
            - **Best Performing Model:** {best_model_name} (RMSE: {best_model_error:.4f})
            - **Combined Forecast:** Weights - Holt-Winters: {hw_weight:.2f}, ARIMA: {arima_weight:.2f}
            - **Trend Prediction:** {'Upward' if combined_future_forecast.mean() > time_series_data.iloc[-1] else 'Downward'} trend predicted
            - **Recommendation:** {'Increase monitoring' if combined_future_forecast.mean() > time_series_data.iloc[-1] else 'Maintain current operations'}
            """)

# Final note
st.markdown("---")
st.markdown("*Note: Forecasts are probabilistic estimates - actual results may vary. Update models regularly with new data.*")
