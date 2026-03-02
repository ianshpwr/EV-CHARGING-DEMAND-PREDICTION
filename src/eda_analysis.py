import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


# ------------------------------------------------
# Historical Trend Chart
# ------------------------------------------------
def plot_historical_trend(station_df, selected_date=None, prediction=None):

    plot_df = station_df.tail(60).copy()
    plot_df["rolling_7"] = plot_df["daily_kwh"].rolling(7).mean()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["daily_kwh"],
        mode="lines",
        name="Daily Demand"
    ))

    fig.add_trace(go.Scatter(
        x=plot_df["date"],
        y=plot_df["rolling_7"],
        mode="lines",
        name="7-Day Rolling Avg"
    ))

    if prediction is not None and selected_date is not None:
        fig.add_trace(go.Scatter(
            x=[pd.to_datetime(selected_date)],
            y=[prediction],
            mode="markers",
            marker=dict(size=12, color="red"),
            name="Forecast"
        ))

    fig.update_layout(
        height=500,
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Energy (kWh)"
    )

    return fig


# ------------------------------------------------
# Monthly Trend
# ------------------------------------------------
def plot_monthly_trend(daily_data):

    monthly = daily_data.groupby(
        daily_data["date"].dt.month
    )["daily_kwh"].mean().reset_index()

    monthly.columns = ["Month", "Avg_kWh"]

    fig = px.line(
        monthly,
        x="Month",
        y="Avg_kWh",
        markers=True
    )

    fig.update_layout(template="plotly_dark")

    return fig


# ------------------------------------------------
# Weekly Heatmap
# ------------------------------------------------
def plot_weekday_heatmap(station_df):

    heat_df = station_df.copy()
    heat_df["Weekday"] = heat_df["date"].dt.day_name()

    pivot = heat_df.pivot_table(
        values="daily_kwh",
        index="Weekday",
        aggfunc="mean"
    ).reindex([
        "Monday","Tuesday","Wednesday","Thursday",
        "Friday","Saturday","Sunday"
    ])

    fig = px.imshow(
        pivot,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Turbo"
    )

    fig.update_layout(template="plotly_dark")

    return fig


# ------------------------------------------------
# Top Stations
# ------------------------------------------------
def plot_top_stations(daily_data):

    top_stations = daily_data.groupby("stationID")["daily_kwh"] \
        .mean().sort_values(ascending=False).head(10)

    fig = px.bar(
        x=top_stations.index,
        y=top_stations.values,
        labels={"x":"Station","y":"Avg Daily kWh"}
    )

    fig.update_layout(template="plotly_dark")

    return fig


# ------------------------------------------------
# Demand Distribution
# ------------------------------------------------
def plot_demand_distribution(station_df):

    fig = px.histogram(
        station_df,
        x="daily_kwh",
        nbins=30
    )

    fig.update_layout(template="plotly_dark")

    return fig


# ------------------------------------------------
# System-Wide Trend
# ------------------------------------------------
def plot_system_trend(daily_data):

    system_trend = daily_data.groupby("date")["daily_kwh"].sum().reset_index()

    fig = px.line(
        system_trend,
        x="date",
        y="daily_kwh"
    )

    fig.update_layout(template="plotly_dark")

    return fig