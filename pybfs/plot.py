# -*- coding: utf-8 -*-
"""Plotting functions for PyBFS

Functions for visualizing baseflow separation results and forecasts.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def plot_baseflow_simulation(streamflow, tmp, title="Baseflow Simulation"):
    """Plots observed streamflow vs simulated baseflow from PyBFS

    Creates a time series plot comparing observed total streamflow with simulated
    baseflow. Useful for visualizing baseflow separation results and assessing
    model performance.

    Parameters
    ----------
    streamflow : pd.DataFrame
        DataFrame containing observed data with columns Date (datetime) and Streamflow
        (observed streamflow m³/day)
    tmp : pd.DataFrame
        Output from PyBFS() containing flow components with column Baseflow (simulated
        baseflow component m³/day)
    title : str, optional
        Plot title, default is "Baseflow Simulation"

    Returns
    -------
    pd.DataFrame
        DataFrame used for plotting with columns date (datetime), streamflow (observed
        streamflow m³/s converted from m³/day), and baseflow (simulated baseflow m³/s
        converted from m³/day)

    Notes
    -----
    Streamflow values are converted from m³/day to m³/s by dividing by 86400 seconds/day.
    Black line shows total observed streamflow. Green line shows simulated baseflow component.
    Displays plot using matplotlib.

    Examples
    --------
    >>> results = bfs(streamflow_data, SBT, basin_char, gw_hyd, flow)
    >>> plot_df = plot_baseflow_simulation(streamflow_data, results)
    """
    # Prepare DataFrame for plotting
    df = pd.DataFrame({
        "date": pd.to_datetime(streamflow["Date"]),
        "streamflow": streamflow["Streamflow"],
        "baseflow": tmp["Baseflow"]
    })

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot observed streamflow (converted from m³/day to m³/s)
    ax.plot(df["date"], df["streamflow"] / 86400, color="black",
            label="Streamflow", linewidth=1)

    # Plot simulated baseflow (converted similarly)
    ax.plot(df["date"], df["baseflow"] / 86400, color="green",
            label="PyBFS", linewidth=1.5)

    # Labels and formatting
    ax.set_xlabel("Date", fontsize=16)
    ax.set_ylabel("Flow (cms)", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.legend(loc="upper right", fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=14)

    # Date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()

    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

    return df


def plot_forecast_baseflow(forecast_data):
    """Plot baseflow forecast time series

    Creates a time series plot of forecasted baseflow. Shows how baseflow
    is expected to evolve over the forecast period based on drainage from
    storage reservoirs.

    Parameters
    ----------
    forecast_data : pd.DataFrame
        Forecast results from forecast() function with columns Date (datetime of forecast)
        and Baseflow (forecasted baseflow m³/day)

    Notes
    -----
    Baseflow values are converted from m³/day to m³/s by dividing by 86400 seconds/day.
    Green line shows forecasted baseflow. Displays plot using matplotlib. No observed
    streamflow is shown (forecast period has no observations).

    Examples
    --------
    >>> forecast_result = forecast(forecast_df, SBT, basin_char, gw_hyd, flow, ini)
    >>> plot_forecast_baseflow(forecast_result)
    """
    fig, axs = plt.subplots(figsize=(9, 5))
    date = pd.to_datetime(forecast_data["Date"])
    axs.plot(date, forecast_data['Baseflow']/86400, color='green', label='PyBFS Baseflow', linewidth=1.5)

    # Add legend
    axs.legend(loc='upper right', fontsize=13)

    # Set title and axis labels
    axs.set_title(f"Baseflow Forecast", fontsize=18)
    axs.set_xlabel('Date', fontsize=16)
    axs.set_ylabel('Flow (cms)', fontsize=16)

    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Tick label font sizes
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)

    plt.show()


def plot_forecast_baseflow_streamflow(forecast_data, streamflow):
    """Plot forecast baseflow with observed streamflow for comparison

    Creates a time series plot comparing forecasted baseflow against observed
    streamflow. Useful for validating forecast performance when observations
    are available for the forecast period.

    Parameters
    ----------
    forecast_data : pd.DataFrame
        Forecast results from forecast() function with columns Date (datetime of forecast)
        and Baseflow (forecasted baseflow m³/day)
    streamflow : pd.DataFrame
        Observed streamflow data for the forecast period with columns Date (datetime of
        observations) and Streamflow (observed streamflow m³/day)

    Notes
    -----
    Flow values are converted from m³/day to m³/s by dividing by 86400 seconds/day.
    Blue line shows observed total streamflow. Green line shows forecasted baseflow.
    Displays plot using matplotlib. Used for forecast validation when observations become available.

    Examples
    --------
    >>> forecast_result = forecast(forecast_df, SBT, basin_char, gw_hyd, flow, ini)
    >>> # Get observed data for same period
    >>> obs_data = streamflow_data[(streamflow_data['Date'] >= '2018-10-01') &
    ...                             (streamflow_data['Date'] <= '2018-11-30')]
    >>> plot_forecast_baseflow_streamflow(forecast_result, obs_data)
    """
    fig, axs = plt.subplots(figsize=(9, 5))
    date = pd.to_datetime(forecast_data["Date"])

    axs.plot(date, streamflow['Streamflow']/86400, color='blue', label='USGS Streamflow', linewidth=1.5)
    axs.plot(date, forecast_data['Baseflow']/86400, color='green', label='PyBFS Baseflow', linewidth=1.5)

    # Add legend
    axs.legend(loc='upper right', fontsize=13)

    # Set title and axis labels
    axs.set_title(f"Baseflow Forecast", fontsize=18)
    axs.set_xlabel('Date', fontsize=16)
    axs.set_ylabel('Flow (cms)', fontsize=16)

    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()

    # Tick label font sizes
    axs.tick_params(axis='x', labelsize=14)
    axs.tick_params(axis='y', labelsize=14)
    #end

    plt.show()

