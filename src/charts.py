"""
src/charts.py
=============
All Plotly visualisation functions — pure chart builders, no side effects.

Each function accepts data and returns a go.Figure ready for st.plotly_chart().
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Consistent dark theme applied to all charts
_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", size=13),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ------------------------------------------------------------------
# 1. Historical Trend  (single station, last 60 days + forecast point)
# ------------------------------------------------------------------
def plot_historical_trend(
    station_df: pd.DataFrame,
    selected_date=None,
    prediction: float | None = None,
) -> go.Figure:
    """60-day daily demand line with 7-day rolling average and optional forecast marker."""
    if station_df.empty:
        return go.Figure()

    # 1 & 3: Ensure date format and filter last 60 days correctly
    df = station_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    max_date = df["date"].max()
    recent_df = df[df["date"] >= max_date - pd.Timedelta(days=60)]

    # 2: Data is already aggregated daily in station_df (daily_kwh)
    # We just need to map it to 'kwh' for the plot
    daily = recent_df[["date", "daily_kwh"]].copy()
    daily.columns = ["date", "kwh"]

    # 5: Handle missing days
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max())
    daily = daily.set_index("date").reindex(full_dates, fill_value=0).rename_axis("date").reset_index()

    # Calculate rolling avg on the contiguous dataset
    daily["rolling_7"] = daily["kwh"].rolling(7, min_periods=1).mean()

    # 4: Plot clean line chart using px.line
    fig = px.line(
        daily, 
        x="date", 
        y="kwh", 
        title="Historical Demand — Last 60 Days",
        labels={"kwh": "Daily Demand (kWh)", "date": "Date"},
        color_discrete_sequence=["#7EB8F7"]
    )
    
    # 7-Day Avg trace
    fig.add_trace(go.Scatter(
        x=daily["date"],
        y=daily["rolling_7"],
        mode="lines",
        name="7-Day Avg",
        line=dict(color="#F7A07E", width=2, dash="dot"),
    ))

    # 6: Fix prediction overlay
    if prediction is not None and selected_date is not None:
        target_ts = pd.to_datetime(selected_date)
        past_data = daily[daily["date"] < target_ts]

        if not past_data.empty:
            last_date = past_data["date"].iloc[-1]
            last_val = past_data["kwh"].iloc[-1]
            
            fig.add_trace(go.Scatter(
                x=[last_date, target_ts],
                y=[last_val, prediction],
                mode="lines+markers",
                name="Forecast Trend",
                line=dict(color="#FFD700", width=2, dash="dash"),
                marker=dict(size=10, color="#FFD700", symbol="star"),
            ))
        else:
            fig.add_trace(go.Scatter(
                x=[target_ts],
                y=[prediction],
                mode="markers",
                name="Forecast",
                marker=dict(size=14, color="#FFD700", symbol="star"),
            ))

    fig.update_layout(
        height=400,
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ------------------------------------------------------------------
# 2. Monthly Average Demand  (system-wide)
# ------------------------------------------------------------------
def plot_monthly_trend(daily_data: pd.DataFrame) -> go.Figure:
    """Bar chart of average daily demand per calendar month."""
    monthly = (
        daily_data
        .groupby(daily_data["date"].dt.month)["daily_kwh"]
        .mean()
        .reset_index()
    )
    monthly.columns = ["Month", "Avg_kWh"]

    month_names = {
        1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
        7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"
    }
    monthly["Month_Name"] = monthly["Month"].map(month_names)

    fig = px.bar(
        monthly, x="Month_Name", y="Avg_kWh",
        labels={"Month_Name": "Month", "Avg_kWh": "Avg Daily kWh"},
        color="Avg_kWh",
        color_continuous_scale="Teal",
    )
    fig.update_layout(height=350, coloraxis_showscale=False, **_LAYOUT_DEFAULTS)
    return fig


# ------------------------------------------------------------------
# 3. Weekday Heatmap  (single station)
# ------------------------------------------------------------------
def plot_weekday_heatmap(station_df: pd.DataFrame) -> go.Figure:
    """Heatmap of average daily kWh by weekday for the selected station."""
    heat_df             = station_df.copy()
    heat_df["Weekday"]  = heat_df["date"].dt.day_name()

    pivot = heat_df.pivot_table(
        values="daily_kwh", index="Weekday", aggfunc="mean"
    ).reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="Turbo",
        labels=dict(color="Avg kWh"),
    )
    fig.update_layout(height=300, **_LAYOUT_DEFAULTS)
    return fig


# ------------------------------------------------------------------
# 4. Top 10 Stations by Average Daily Demand
# ------------------------------------------------------------------
def plot_top_stations(daily_data: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top-N highest-demand stations."""
    top = (
        daily_data.groupby("stationID")["daily_kwh"]
        .mean()
        .sort_values(ascending=True)
        .tail(top_n)
    )

    fig = go.Figure(go.Bar(
        x=top.values,
        y=top.index,
        orientation="h",
        marker=dict(
            color=top.values,
            colorscale="Plasma",
            showscale=False,
        ),
    ))
    fig.update_layout(
        height=380,
        xaxis_title="Avg Daily kWh",
        yaxis_title="Station",
        **_LAYOUT_DEFAULTS,
    )
    return fig


# ------------------------------------------------------------------
# 5. Demand Distribution Histogram  (single station)
# ------------------------------------------------------------------
def plot_demand_distribution(station_df: pd.DataFrame) -> go.Figure:
    """Histogram of daily kWh values for the selected station."""
    fig = px.histogram(
        station_df, x="daily_kwh", nbins=30,
        labels={"daily_kwh": "Daily kWh"},
        color_discrete_sequence=["#7EB8F7"],
    )
    fig.update_layout(height=320, **_LAYOUT_DEFAULTS)
    return fig


# ------------------------------------------------------------------
# 6. System-Wide Daily Demand Trend
# ------------------------------------------------------------------
def plot_system_trend(daily_data: pd.DataFrame) -> go.Figure:
    """Line chart of total daily kWh summed across all stations."""
    system = daily_data.groupby("date")["daily_kwh"].sum().reset_index()

    fig = px.line(
        system, x="date", y="daily_kwh",
        labels={"daily_kwh": "Total System kWh", "date": "Date"},
        color_discrete_sequence=["#F7A07E"],
    )
    fig.update_layout(height=350, **_LAYOUT_DEFAULTS)
    return fig


# ------------------------------------------------------------------
# 7. Geospatial Map  (system-wide load)
# ------------------------------------------------------------------
def plot_system_map(predictions_dict: dict) -> go.Figure:
    """Scatter mapbox of stations colored/sized by predicted load."""
    import numpy as np
    
    if not predictions_dict:
        return go.Figure()
        
    stations = list(predictions_dict.keys())
    loads = [max(v, 0.1) for v in predictions_dict.values()] # ensure positive for size
    
    # Generate mock coordinates around Caltech (34.1377, -118.1253)
    np.random.seed(42)
    lats = 34.1377 + np.random.uniform(-0.005, 0.005, len(stations))
    lons = -118.1253 + np.random.uniform(-0.005, 0.005, len(stations))
    
    df_map = pd.DataFrame({
        "Station": stations,
        "Load (kWh)": loads,
        "lat": lats,
        "lon": lons
    })
    
    fig = px.scatter_mapbox(
        df_map, 
        lat="lat", 
        lon="lon", 
        color="Load (kWh)", 
        size="Load (kWh)",
        color_continuous_scale="Plasma",
        hover_name="Station", 
        hover_data={"lat": False, "lon": False, "Load (kWh)": ":.1f"},
        zoom=14, 
        center={"lat": 34.1377, "lon": -118.1253},
        mapbox_style="carto-darkmatter"
    )
    
    fig.update_layout(
        height=450,
        margin={"r":0,"t":0,"l":0,"b":0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig

