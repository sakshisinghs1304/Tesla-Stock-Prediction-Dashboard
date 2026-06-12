import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go

# PAGE CONFIGURATIONs

st.set_page_config(
    page_title="Tesla Stock Price Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>

[data-testid="stMetricValue"] {
    font-size: 28px;
    font-weight: bold;
}

[data-testid="stSidebar"] {
    background-color: #0E1117;
}

h1 {
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)

# LOAD DATASET

@st.cache_data
def load_data():
    return pd.read_csv("TESLA.csv")


# LOAD MODEL & SCALER

@st.cache_resource
def load_assets():
    model = load_model("best_tesla_stock_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

df = load_data()
model, scaler = load_assets()

# SIDEBAR

st.sidebar.title("📈 Tesla Stock Prediction")
st.sidebar.markdown(
    """
    ### Deep Learning Dashboard
    
    **Models Used**
    - SimpleRNN
    - LSTM
    - Bidirectional LSTM
    
    **Best Model**
    - Optimized SimpleRNN
    """
)

st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    [
        "Forecasting Engine",
        "Market Analytics",
        "Deep Learning Insights",
        "Model Performance",
        "Business Intelligence",
        "About Project"
    ]
)

st.sidebar.markdown("---")

st.sidebar.success("Best Model: Optimized SimpleRNN")
st.sidebar.metric("R² Score", "0.9557")


# FORECASTING ENGINE

if page == "Forecasting Engine":

    st.title("Tesla Stock Price Prediction")

    st.markdown(
        """
        ### Financial Forecasting & Deep Learning Analytics Platform

        Predict Tesla stock prices using an optimized Deep Learning model
        and generate interactive future forecasts.
        """
    )

    st.markdown("---")

    latest_close = float(df["Close"].iloc[-1])

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Current Tesla Price",
            f"${latest_close:.2f}"
        )

    with col2:
        st.metric(
            "Forecast Horizon",
            "1 - 30 Days"
        )

    st.markdown("---")

    st.subheader("Future Price Forecasting")

    forecast_days = st.slider(
        "Select Forecast Horizon (Days)",
        min_value=1,
        max_value=30,
        value=10
    )

    investment = st.number_input(
        "Investment Amount ($)",
        min_value=100,
        value=1000,
        step=100
    )

    generate = st.button(
        "Generate Forecast"
    )

    if generate:

        close_prices = df["Close"].values

        last_60 = close_prices[-60:]

        scaled_input = scaler.transform(
            last_60.reshape(-1, 1)
        )

        sequence = scaled_input.copy()

        predictions = []

        for _ in range(forecast_days):

            x_input = sequence.reshape(
                1,
                60,
                1
            )

            pred = model.predict(
                x_input,
                verbose=0
            )

            predictions.append(
                pred[0][0]
            )

            sequence = np.vstack(
                [sequence[1:], pred]
            )

        forecast = scaler.inverse_transform(
            np.array(predictions).reshape(-1, 1)
        )

        day1_price = forecast[
            min(0, len(forecast)-1)
        ][0]

        day5_price = forecast[
            min(4, len(forecast)-1)
        ][0]

        day10_price = forecast[
            min(9, len(forecast)-1)
        ][0]

        selected_price = forecast[-1][0]

        st.markdown("---")

        st.subheader(
            "Investment Growth Simulator"
        )

        r1, r2, r3, r4 = st.columns(4)

        with r1:

            value_1 = (
                investment *
                day1_price /
                latest_close
            )

            st.metric(
                "1 Day",
                f"${value_1:.2f}"
            )

        with r2:

            value_5 = (
                investment *
                day5_price /
                latest_close
            )

            st.metric(
                "5 Days",
                f"${value_5:.2f}"
            )

        with r3:

            value_10 = (
                investment *
                day10_price /
                latest_close
            )

            st.metric(
                "10 Days",
                f"${value_10:.2f}"
            )

        with r4:
            best_care_price = forecast.max()
            
            value_selected=(
                investment *
                selected_price /
                latest_close
            )

            st.metric(
                "Potential Value",
                f"${value_selected:.2f}"
            )

        st.markdown("---")

        st.subheader(
            "Forecast Trend Analysis"
        )

        actual_prices = df["Close"].tail(30).values

        actual_df = pd.DataFrame({
            "Index": list(range(1, 31)),
            "Price": actual_prices,
            "Type": ["Actual"] * 30
        })

        forecast_df = pd.DataFrame({
            "Index": list(
                range(
                    31,
                    31 + len(forecast)
                )
            ),
            "Price": forecast.flatten(),
            "Type": ["Forecast"] * len(forecast)
        })

        combined_df = pd.concat(
            [actual_df, forecast_df],
            ignore_index=True
        )

        fig = px.line(
            combined_df,
            x="Index",
            y="Price",
            color="Type",
            markers=True
        )

        fig.add_vline(
            x=30,
            line_width=3,
            line_dash="dash",
            line_color="orange"
        )

        fig.add_annotation(
            x=30,
            y=combined_df["Price"].max(),
            text="Forecast Starts",
            showarrow=False
        )
        
        fig.add_hline(
            y=forecast.max(),
            line_dash="dot",
            line_color="green",
            annotation_text="Best Case Scenario"
        )

        fig.update_traces(
            line=dict(width=4)
        )

        fig.update_layout(
            template="plotly_dark",
            height=700,
            title="Actual vs Forecasted Tesla Stock Prices",
            title_x=0.5,
            xaxis_title="Timeline",
            yaxis_title="Stock Price ($)",
            hovermode="x unified"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.markdown("---")

        st.subheader(
            "AI Market Insight"
        )

        final_change = (
            (
                selected_price -
                latest_close
            )
            / latest_close
        ) * 100

        if final_change > 5:

            st.success(
                f"""
                The model forecasts a strong upward trend
                over the selected {forecast_days}-day horizon.

                Estimated growth:
                {final_change:.2f}%.
                """
            )

        elif final_change > 0:

            st.info(
                f"""
                The model forecasts moderate positive growth
                over the selected horizon.

                Estimated growth:
                {final_change:.2f}%.
                """
            )

        else:

            st.warning(
                f"""
                The model forecasts a short-term correction
                over the selected horizon.

                Estimated movement:
                {final_change:.2f}%.
                """
            )
            
# MARKET ANALYTICS

elif page == "Market Analytics":

    st.title("Market Analytics")

    
    # Closing Price Trend

    st.subheader("Tesla Closing Price Trend")

    fig = px.line(
        df,
        x="Date",
        y="Close",
        title="Historical Closing Price"
    )
    
    fig.update_traces(
    line=dict(width=5)
    )
    
    fig.update_layout(
        template="plotly_white",
        height=500
    )
    
    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "Tesla stock demonstrates significant long-term growth with periods of elevated volatility."
    )

    # -----------------------------------------
    # Volume Trend
    # -----------------------------------------

    st.subheader("Trading Volume Trend")

    fig = px.line(
        df,
        x="Date",
        y="Volume",
        title="Historical Trading Volume"
    )

    fig.update_layout(
        template="plotly_dark",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.info(
        "High trading volume often coincides with major price movements and investor activity."
    )

    # -----------------------------------------
    # Moving Averages
    # -----------------------------------------

    st.subheader("Moving Average Analysis")

    analytics_df = df.copy()

    analytics_df["MA_7"] = (
        analytics_df["Close"]
        .rolling(7)
        .mean()
    )

    analytics_df["MA_30"] = (
        analytics_df["Close"]
        .rolling(30)
        .mean()
    )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=analytics_df["Date"],
            y=analytics_df["Close"],
            name="Close Price"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=analytics_df["Date"],
            y=analytics_df["MA_7"],
            name="7-Day MA"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=analytics_df["Date"],
            y=analytics_df["MA_30"],
            name="30-Day MA"
        )
    )

    fig.update_layout(
        title="7-Day vs 30-Day Moving Average",
        template="plotly_white",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Moving averages smooth market noise and help identify trend reversals."
    )

    # Daily Returns Distribution

    st.subheader("Daily Returns Distribution")

    analytics_df["Daily_Return"] = (
        analytics_df["Close"]
        .pct_change()
    )

    fig = px.histogram(
        analytics_df,
        x="Daily_Return",
        nbins=50,
        title="Distribution of Daily Returns"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
        "Most daily returns cluster around zero while extreme gains and losses occur less frequently."
    )

    # Closing Price Distribution

    st.subheader("Closing Price Distribution")

    fig = px.histogram(
        df,
        x="Close",
        nbins=50,
        title="Tesla Closing Price Distribution"
    )

    fig.update_layout(
        template="plotly_dark",
        height=600
        )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.info(
    """
    The distribution illustrates the historical spread of Tesla
    closing prices and highlights periods of significant price growth.
    """
    )

# DEEP LEARNING INSIGHTS


elif page == "Deep Learning Insights":

    st.title("Deep Learning Insights")

    st.markdown(
        """
        This section summarizes the complete Deep Learning pipeline
        used for Tesla stock price prediction.
        """
    )

 
    # PIPELINE SUMMARY


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Target Variable",
            "Close Price"
        )

    with col2:
        st.metric(
            "Sequence Length",
            "60 Days"
        )

    with col3:
        st.metric(
            "Scaling Method",
            "MinMaxScaler"
        )

    st.markdown("---")

    # -----------------------------------------
    # MODEL WORKFLOW
    # -----------------------------------------

    st.subheader("Training Configuration")

    training_df = pd.DataFrame({
      "Parameter": [
        "Target Variable",
        "Sequence Length",
        "Scaling Technique",
        "Loss Function",
        "Optimizer",
        "Forecast Horizons"
      ],
      "Value": [
        "Close Price",
        "60 Days",
        "MinMaxScaler",
        "MSE",
        "Adam",
        "1 / 5 / 10 Days"
      ]
    })

    st.dataframe(
        training_df,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Model Selection Summary")

    st.success("""
Optimized SimpleRNN achieved the best overall forecasting performance.

• Lowest RMSE

• Lowest MAE

• Highest R² Score

Therefore it was selected as the final deployment model.
""")

    st.markdown("---")

    # -----------------------------------------
    # MODEL ARCHITECTURES
    # -----------------------------------------

    st.subheader("Models Developed")

    architecture_df = pd.DataFrame({
        "Model": [
            "SimpleRNN",
            "LSTM",
            "Bidirectional LSTM"
        ],
        "Purpose": [
            "Sequential Pattern Learning",
            "Long-Term Dependency Learning",
            "Forward & Backward Sequence Learning"
        ]
    })

    st.dataframe(
        architecture_df,
        use_container_width=True
    )

    st.markdown("---")

    # -----------------------------------------
    # WHY SIMPLE RNN WON
    # -----------------------------------------

    st.subheader("Why Optimized SimpleRNN Was Selected")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "RMSE",
            "0.020157"
        )

    with col2:
        st.metric(
            "MAE",
            "0.012814"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.955735"
        )

    st.success(
        """
        The Optimized SimpleRNN achieved the lowest prediction error
        and the highest R² score among all evaluated models.
        Therefore it was selected as the final deployment model.
        """
    )

    st.markdown("---")

    # -----------------------------------------
    # FORECASTING CAPABILITY
    # -----------------------------------------

    st.subheader("Forecasting Capability")

    forecast_df = pd.DataFrame({
        "Forecast Horizon": [
            "1 Day Ahead",
            "5 Days Ahead",
            "10 Days Ahead"
        ],
        "Supported": [
            "Yes",
            "Yes",
            "Yes"
        ]
    })

    st.dataframe(
        forecast_df,
        use_container_width=True
    )

    st.info(
        """
        The deployed model supports short-term and medium-term
        forecasting for Tesla stock prices.
        """
    )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title("Model Performance Comparison")

    st.markdown("""
### Benchmarking Deep Learning Models

Three Deep Learning architectures were evaluated and compared
using RMSE, MAE, and R² Score.
""")

    performance = pd.DataFrame({
        "Model": [
            "Optimized SimpleRNN",
            "LSTM",
            "Bidirectional LSTM"
        ],
        "RMSE": [
            0.020157,
            0.029525,
            0.023417
        ],
        "MAE": [
            0.012814,
            0.020783,
            0.015783
        ],
        "R² Score": [
            0.955735,
            0.905027,
            0.940259
        ]
    })

    st.subheader("Model Comparison Table")

    st.dataframe(
        performance,
        use_container_width=True
    )

    st.markdown("---")

    
    # R2 COMPARISON

    st.subheader("R² Score Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="R² Score",
        text="R² Score",
        title="Model Accuracy Comparison"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================
    # RMSE COMPARISON
    # =====================================

    st.subheader("RMSE Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Root Mean Squared Error"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================
    # MAE COMPARISON
    # =====================================

    st.subheader("MAE Comparison")

    fig = px.bar(
        performance,
        x="Model",
        y="MAE",
        text="MAE",
        title="Mean Absolute Error"
    )

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    st.subheader("Final Model Selection")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "RMSE",
            "0.020157"
        )

    with col2:
        st.metric(
            "MAE",
            "0.012814"
        )

    with col3:
        st.metric(
            "R² Score",
            "0.955735"
        )

    st.success(
        """
        Final Selected Model: Optimized SimpleRNN

        The Optimized SimpleRNN achieved the lowest RMSE,
        lowest MAE and highest R² score among all models,
        making it the most accurate forecasting model.
        """
    )

    st.markdown("---")

    st.header("🏆 Best Performing Model")

    st.success("""
Optimized SimpleRNN achieved:

• Lowest RMSE

• Lowest MAE

• Highest R² Score

Therefore it was selected as the final deployment model.
""")

# =====================================================
# BUSINESS INTELLIGENCE
# =====================================================

elif page == "Business Intelligence":

    st.title("Business Intelligence")

    st.markdown(
        """
        This section demonstrates how Tesla stock forecasts
        can be used in real-world financial applications.
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("Automated Trading")

        st.info(
            """
            Forecasted stock prices can support
            algorithmic buy and sell decisions,
            enabling automated trading strategies.
            """
        )

        st.subheader("Risk Management")

        st.info(
            """
            Future price predictions help investors
            estimate downside risk and prepare
            suitable hedging strategies.
            """
        )

    with col2:

        st.subheader("Portfolio Optimization")

        st.info(
            """
            Investors can rebalance portfolios
            based on expected future Tesla stock
            performance.
            """
        )

        st.subheader("Long-Term Investment Planning")

        st.info(
            """
            Deep Learning forecasts assist
            long-term investors in evaluating
            growth opportunities.
            """
        )

    st.markdown("---")

    st.subheader("Business Impact Summary")

    business_df = pd.DataFrame({
        "Application": [
            "Algorithmic Trading",
            "Risk Management",
            "Portfolio Optimization",
            "Investment Planning",
            "Financial Forecasting"
        ],
        "Impact": [
            "Improved Trading Decisions",
            "Reduced Financial Risk",
            "Better Asset Allocation",
            "Data Driven Investments",
            "Future Market Insights"
        ]
    })

    st.dataframe(
        business_df,
        use_container_width=True
    )

    # =====================================================
# ABOUT PROJECT
# =====================================================

elif page == "About Project":

    st.title("About Project")

    st.markdown("""
    ## Tesla Stock Price Prediction Using Deep Learning

    ### Objective
    Predict Tesla stock prices using historical market data and
    Deep Learning techniques.

    ### Models Developed
    - SimpleRNN
    - LSTM
    - Bidirectional LSTM

    ### Final Model
    Optimized SimpleRNN

    ### Deployment
    Streamlit Dashboard

    ### Domain
    Financial Services
    """)

    st.markdown("---")

    st.markdown(
    """
    <div style="
        text-align:center;
        font-size:24px;
        font-weight:bold;
        color:white;
        padding-top:10px;
    ">
        Developed by Sakshi Singh
    </div>

    <div style="
        text-align:center;
        font-size:16px;
        color:gray;
    ">
        Deep Learning Financial Forecasting Project
    </div>
    """,
    unsafe_allow_html=True
)