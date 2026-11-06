
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="AI Alpha Discovery Platform",
    page_icon="📈",
    layout="wide"
)

st.title("AI + Alternative Data Alpha Discovery Platform")
st.caption("Research-First Quantitative Finance | MVP v0.1")

st.sidebar.header("Research Controls")

data_source = st.sidebar.radio(
    "Select Data Source",
    ["Synthetic Demo Data", "Upload CSV"]
)

train_ratio = st.sidebar.slider(
    "Training Data Ratio",
    0.50, 0.90, 0.70
)

top_fraction = st.sidebar.slider(
    "Portfolio Selection %",
    0.05, 0.50, 0.20
)

cost_bps = st.sidebar.number_input(
    "Transaction Cost (BPS)",
    min_value=0.0,
    max_value=100.0,
    value=10.0
)


@st.cache_data
def generate_demo_data():

    rng = np.random.default_rng(42)

    dates = pd.bdate_range(
        "2019-01-01",
        periods=1250
    )

    tickers = [
        "ALPHA",
        "BETA",
        "GAMMA",
        "DELTA",
        "OMEGA"
    ]

    records = []

    for ticker in tickers:

        returns = rng.normal(
            0.0003, 0.01, len(dates)
        )

        prices = 100 * np.exp(
            np.cumsum(returns)
        )

        volume = rng.lognormal(
            13, 0.25, len(dates)
        )

        alternative_activity = (
            100 + rng.normal(
                0, 10, len(dates)
            )
        )

        for i, date in enumerate(dates):

            records.append({
                "date": date,
                "ticker": ticker,
                "close": prices[i],
                "volume": volume[i],
                "alt_activity": alternative_activity[i]
            })

    return pd.DataFrame(records)


if data_source == "Upload CSV":

    uploaded = st.file_uploader(
        "Upload financial CSV",
        type=["csv"]
    )

    if uploaded is None:
        st.info("Upload your CSV to continue.")
        st.stop()

    raw = pd.read_csv(uploaded)

else:

    raw = generate_demo_data()


required = {"date", "ticker", "close"}

if not required.issubset(raw.columns):

    st.error(
        "Required columns: date, ticker, close"
    )
    st.stop()


raw["date"] = pd.to_datetime(
    raw["date"], errors="coerce"
)

raw["close"] = pd.to_numeric(
    raw["close"], errors="coerce"
)

raw = raw.dropna(
    subset=["date", "ticker", "close"]
)

raw = raw.sort_values(
    ["ticker", "date"]
)

if "volume" not in raw.columns:
    raw["volume"] = np.nan

if "alt_activity" not in raw.columns:
    raw["alt_activity"] = np.nan


def create_features(df):

    df = df.copy()

    grouped = df.groupby("ticker")

    df["return_1d"] = (
        grouped["close"].pct_change()
    )

    df["momentum_5d"] = (
        grouped["close"].pct_change(5)
    )

    df["momentum_20d"] = (
        grouped["close"].pct_change(20)
    )

    df["volatility_20d"] = (
        df.groupby("ticker")["return_1d"]
        .transform(
            lambda x: x.rolling(
                20, min_periods=10
            ).std()
        )
    )

    df["volume_change"] = (
        df.groupby("ticker")["volume"]
        .pct_change()
    )

    df["alt_change"] = (
        df.groupby("ticker")["alt_activity"]
        .pct_change()
    )

    df[["volume_change", "alt_change"]] = (
        df[["volume_change", "alt_change"]]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    df["target"] = (
        df.groupby("ticker")["close"]
        .shift(-1) / df["close"] - 1
    )

    features = [
        "momentum_5d",
        "momentum_20d",
        "volatility_20d",
        "volume_change",
        "alt_change"
    ]

    df = df.replace(
        [np.inf, -np.inf], np.nan
    )

    research_data = df.dropna(
        subset=features + ["target"]
    ).copy()

    return research_data, features


data, features = create_features(raw)

if data["date"].nunique() < 60:
    st.error("Please upload at least 60 trading dates.")
    st.stop()

dates = sorted(data["date"].unique())

split_date = dates[
    int(len(dates) * train_ratio)
]

train = data[
    data["date"] < split_date
].copy()

test = data[
    data["date"] >= split_date
].copy()

if len(train) < 50 or len(test) < 20:
    st.error("Insufficient training or test observations.")
    st.stop()

X_train = train[features]
y_train = train["target"]

X_test = test[features]
y_test = test["target"]


model = RandomForestRegressor(
    n_estimators=150,
    max_depth=5,
    min_samples_leaf=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

test["prediction"] = model.predict(X_test)

test["actual_return"] = y_test

test["rank"] = (
    test.groupby("date")["prediction"]
    .rank(pct=True)
)

test["selected"] = (
    test["rank"] >= (1 - top_fraction)
)

daily = test.groupby("date").apply(
    lambda x: (
        x.loc[x["selected"], "actual_return"].mean()
        if x["selected"].any()
        else 0.0
    ),
    include_groups=False
).to_frame("gross_return")

daily["net_return"] = (
    daily["gross_return"] - cost_bps / 10000
)

daily["equity_curve"] = (
    1 + daily["net_return"]
).cumprod()

daily["benchmark"] = (
    1 + test.groupby("date")["actual_return"].mean()
).cumprod()

daily["drawdown"] = (
    daily["equity_curve"] /
    daily["equity_curve"].cummax() - 1
)

std = daily["net_return"].std()

sharpe = (
    daily["net_return"].mean() /
    std * np.sqrt(252)
    if std > 0 else 0
)

total_return = (
    daily["equity_curve"].iloc[-1] - 1
)

max_drawdown = daily["drawdown"].min()

mae = mean_absolute_error(
    y_test, test["prediction"]
)


st.success("Research pipeline executed successfully.")

st.warning(
    "Synthetic data is illustrative. "
    "This prototype does not establish real investment alpha."
)

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Test Observations",
    f"{len(test):,}"
)

c2.metric(
    "Net Cumulative Return",
    f"{total_return:.2%}"
)

c3.metric(
    "Annualised Sharpe",
    f"{sharpe:.2f}"
)

c4.metric(
    "Maximum Drawdown",
    f"{max_drawdown:.2%}"
)


tab1, tab2, tab3, tab4 = st.tabs([
    "Dashboard",
    "Alpha Signals",
    "Backtesting",
    "Research Methodology"
])


with tab1:

    st.subheader("Portfolio Performance")

    chart_data = daily[
        ["equity_curve", "benchmark"]
    ]

    st.plotly_chart(
        px.line(
            chart_data,
            title="Illustrative Out-of-Sample Performance"
        ),
        use_container_width=True
    )


with tab2:

    st.subheader("AI Alpha Predictions")

    latest_date = test["date"].max()

    latest = test[
        test["date"] == latest_date
    ].sort_values(
        "prediction",
        ascending=False
    )

    st.dataframe(
        latest[
            [
                "ticker",
                "prediction",
                "actual_return",
                "selected"
            ]
        ],
        use_container_width=True
    )

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })

    st.plotly_chart(
        px.bar(
            importance,
            x="Feature",
            y="Importance",
            title="Model Feature Importance"
        ),
        use_container_width=True
    )

    st.metric("Mean Absolute Error", f"{mae:.5f}")


with tab3:

    st.subheader("Quantitative Backtesting")

    st.line_chart(
        daily[
            ["equity_curve", "benchmark"]
        ]
    )

    st.dataframe(daily.tail(20))

    st.download_button(
        "Download Predictions",
        test.to_csv(index=False),
        "alpha_predictions.csv",
        "text/csv"
    )


with tab4:

    st.subheader("Research Architecture")

    st.markdown("""
    1. Market and Economic Context
    2. Alternative Data Sourcing
    3. Data Engineering
    4. Feature and Signal Engineering
    5. AI and Quantitative Intelligence
    6. Alpha Validation and Backtesting
    7. Portfolio and Risk Management
    8. Governance and Operations
    """)

    st.write("Model: Random Forest Regression")
    st.write("Validation: Chronological train/test split")
    st.write("Target: Next-observation return")
    st.write("Features: Momentum, volatility, volume and alternative activity")

    st.warning(
        "This MVP omits full point-in-time data controls, "
        "survivorship-bias controls, multiple-testing corrections, "
        "market impact, capacity and live execution."
    )


st.divider()

st.caption(
    "AI + Alternative Data Alpha Discovery Platform | "
    "Harsh Singh | Research Prototype"
)
