
# =============================================================================
# climoDL.py 
#
# Author: mdgrossi
# Modified: Nov 1, 2023
#
# This script retrieves NOAA CO-OPS observational data, both atmospheric and
# oceanic, for the specified station. If historical data already exists
# locally, it is updated with the most recently available observations.
#
# TO EXECUTE:
# python climoDL.py -s "Virginia Key, FL" -i "8723214" -u "english" -t "lst" -d "MHHW" --hr 3 --day 2 
#
# =============================================================================

import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import argparse
import os
import numpy as np
import pandas as pd
from pyclimo import Data
from noaa_coops import Station
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# FUNCTIONS
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Function control parameters.',
                        prog='climoDL',
                        usage='%(prog)s [arguments]')
    parser.add_argument('-s', '--station', metavar='station', type=str,
                        help='Desired name of station. Used for saving data.',
                        default=None)
    parser.add_argument('-i', '--id', metavar='stationid', type=str,
                        help='Tide station number from which to retrieve data.',
                        default=None)
    parser.add_argument('-o', '--outdir', metavar='outdir', type=str,
                        help='Directory to save data to.',
                        default=None)
    parser.add_argument('-u', '--units', metavar='units', type=str,
                      help='Data units, either "metric" or "english".',
                      default='english')
    parser.add_argument('-t', '--timezone', metavar='timezone', type=str,
                      help='Timezone, either "gmt", "lst", or "lst_ldt".',
                      default='lst')
    parser.add_argument('-d', '--datum', metavar='datum', type=str,
                        help='Tidal datum for water level data. Options: '+
                             '"STND", "MHHW", "MHW", "MTL", "MSL", "MLW", '+
                             '"MLLW", "NAVD"',
                        default='MHHW')
    parser.add_argument('--hr', metavar='hr_threshold', type=int,
                        help='Max number of hours of data that can be missing.',
                        default=3)
    parser.add_argument('--day', metavar='day_threshold', type=int,
                        help='Max number of days of data that can be missing.',
                        default=2)
    parser.add_argument('-r', '--redownload', action='store_true',
                      help='Force redownload of historical data.')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Print statuses to screen.')
    return parser.parse_args()

def ploy_fit(data, degree=5, print_coefs=False, plot=False):
    """Fit polynomial curve to data"""
    # Fit curve to data
    y = data.values
    x = np.arange(0, len(y))
    coef = np.polyfit(x, y, degree)
    polyfun = np.poly1d(coef)
    if print_coefs:
        print(f'Coefficients: {coef}')
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,5))
        ax.plot(data, label=data.name)
        ax.plot(polyfun(x), color='red', label=f'{degree}D Polynomial')
        ax.legend(loc='best')
        plt.show()
    else:
        return polyfun(x)

def cos_fit(data, plot=False):
    """Fit cosine curve to data"""
    X = np.arange(0, len(data))/len(data)

    # Initial parameter values
    guess_freq = 1
    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)
    p0 = [guess_freq, guess_amplitude,
          guess_phase, guess_offset]

    # Function to fit
    def my_cos(x, freq, amplitude, phase, offset):
        return np.cos(x * freq + phase) * amplitude + offset

    # Fit curve to data
    fit = curve_fit(my_cos, X, data, p0=p0)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(12,5))

        ax.plot(data, label=data.name)
        ax.plot(fit, color='red', label=f'Cosine fit')

        ax.legend(loc='best')
        plt.show()
    else:
        return my_cos(np.array(X), *fit[0])

def daily_climo(data, var, station, stationid, first_time, last_time,
                scheme='mg',show=False):
    """Create a daily climatology plot for environmental variable 'var'
    from 'data'.
    
    Inputs:
        data: dict, climatological stats dictionary from Data class object
        var: str, one of the available environmental variables in 'data'
        station: str, name of CO-OPS station to include in plot title
        stationid: int, CO-OPS station ID number to include in plot title
        first_time: timestamp of oldest observation to include in plot title
        last_time: timestamp of latest observation to include in plot title
        scheme: str, either 'mg' or 'bm' specifying whether to use M. Grossi's
            color scheme or B. McNoldy's
        show: Bool, display the plot to screen instead of saving to file
    """

    # Dates for x axis
    xdates = pd.date_range(start='2020-01-01',end='2020-12-31', freq='1D')
    df = data[var]
    
    # Color dictionary
    colors = dict(
        mg=dict({
            'Record High Year': 'white',
            'Record High': 'orange',
            'Average High': 'red',
            'Daily Average': 'grey',
            'Average Low': 'purple',
            'Record Low': 'white'}),
        bm=dict({
            'Record High Year': 'white',
            'Record High': 'orange',
            'Average High': 'red',
            'Daily Average': 'grey',
            'Average Low': 'purple',
            'Record Low': 'white'}        
        ))
    
    # Create figure
    fig = go.Figure()

    # Record highs
    # High records this year
    thisYear = pd.to_datetime('today').year
    highRecords = df.loc[df['Record High Year']==thisYear, 'Record High']
    highRecords.index = pd.to_datetime(highRecords.index+'-2020')
    fig.add_trace(
    go.Scatter(
        x=highRecords.index, y=highRecords.values,
        name=f'{pd.to_datetime("today").year} Record'.upper(),
        mode='markers',
        marker=dict(size=6, color='white'),
        hoverinfo='none'
    ))
    fig.add_trace(
    go.Scatter(
        x=xdates, y=df['Record High'],
        name='Record High'.upper(),
        mode='markers',
        marker=dict(size=3, color='orange')
    ))
    # Average highs
    fig.add_trace(
    go.Scatter(
        x=xdates, y=cos_fit(df['Average High']).round(1),
        name='Average High'.upper(),
        marker=dict(size=3, color='red')
    ))
    # Daily average
    fig.add_trace(
    go.Scatter(
        x=xdates, y=cos_fit(df['Daily Average']).round(1),
        name='Daily Average'.upper(),
        marker=dict(size=3, color='grey')
    ))
    # Average lows
    fig.add_trace(
    go.Scatter(
        x=xdates,
        y=cos_fit(df['Average Low']).round(1),
        name='Average Low'.upper(),
        marker=dict(size=3, color='purple')
    ))
    # Record lows
    fig.add_trace(
    go.Scatter(
        x=xdates, y=df['Record Low'],
        name='Record Low'.upper(),
        mode='markers',
        marker=dict(size=3, color='white')
    ))
    # Hover box
    fig.update_traces(
        # mode = 'markers',    
        hoverlabel = dict(bordercolor='white')
    )
    # Plot settings
    fig.update_layout(
        template='plotly_dark',
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=600, width=1000,
        title=dict(text=f'Daily {var} Climatology for {station}'.upper()+
                        '<br><sup>NOAA CO-OPS Site {}, {} - {}</sup>'.format(
                            stationid,
                            first_time.strftime('%m/%d/%Y'),
                            last_time.strftime('%m/%d/%Y')),
                   font=dict(size=20,
                              # family='PT Sans Narrow'
                         )),
        # yaxis = dict(title=f'{var} ({vk.units[var]})'.upper()),
        xaxis = dict(showgrid=False, showspikes=True,
                     dtick='M1', tickformat='%b %d'),
        hovermode='x unified',
        legend=dict(itemsizing='constant'),
        hoverlabel=dict(font_size=12,
                        # font_family="Rockwell"
                       )
    )
    if show:
        fig.show()
    else:
        return fig

def monthly_climo(data, var, station, stationid, first_time, last_time,
                  scheme='mg', show=False):
    """Create a monthly climatology plot for environmental variable 'var'
    from 'data'.
    
    Inputs:
        data: dict, climatological stats dictionary from Data class object
        var: str, one of the available environmental variables in 'data'
        station: str, name of CO-OPS station to include in plot title
        stationid: int, CO-OPS station ID number to include in plot title
        first_time: timestamp of oldest observation to include in plot title
        last_time: timestamp of latest observation to include in plot title
        scheme: str, either 'mg' or 'bm' specifying whether to use M. Grossi's
            color scheme or B. McNoldy's
        show: Bool, display the plot to screen instead of saving to file
    """

    # Dates for x axis
    xdates = pd.date_range(start='2020-01-01',end='2020-12-31', freq='MS')
    df = data[var]
    
    # Color dictionary
    colors = dict(
        mg=dict({
            'Record High Year': 'white',
            'Record High': 'orange',
            'Average High': 'red',
            'Monthly Average': 'grey',
            'Average Low': 'purple',
            'Record Low': 'white'}),
        bm=dict({
            'Record High Year': 'white',
            'Record High': 'orange',
            'Average High': 'red',
            'Monthly Average': 'grey',
            'Average Low': 'purple',
            'Record Low': 'white'}        
        ))
    
    # Create figure
    fig = go.Figure()

    # Record highs
    # High records this year
    thisYear = pd.to_datetime('today').year
    high_records = df.loc[df['Record High Year']==thisYear, 'Record High']
    high_records.index = pd.to_datetime(high_records.index+'-2020')
    fig.add_trace(
    go.Scatter(
        x=high_records.index, y=high_records.values,
        name=f'{pd.to_datetime("today").year} Record'.upper(),
        mode='markers',
        marker=dict(size=6, color='white'),
        hoverinfo='none'
    ))
    fig.add_trace(
    go.Scatter(
        x=xdates, y=df['Record High'],
        name='Record High'.upper(),
        mode='markers',
        marker=dict(size=3, color='orange')
    ))
    # Average highs
    fig.add_trace(
    go.Scatter(
        x=xdates, y=cos_fit(df['Average High']).round(1),
        name='Average High'.upper(),
        marker=dict(size=3, color='red')
    ))
    # Daily average
    fig.add_trace(
    go.Scatter(
        x=xdates, y=cos_fit(df['Monthly Average']).round(1),
        name='Monthly Average'.upper(),
        marker=dict(size=3, color='grey')
    ))
    # Average lows
    fig.add_trace(
    go.Scatter(
        x=xdates,
        y=cos_fit(df['Average Low']).round(1),
        name='Average Low'.upper(),
        marker=dict(size=3, color='purple')
    ))
    # Record lows
    fig.add_trace(
    go.Scatter(
        x=xdates, y=df['Record Low'],
        name='Record Low'.upper(),
        mode='markers',
        marker=dict(size=3, color='white')
    ))
    # Hover box
    fig.update_traces(
        # mode = 'markers',    
        hoverlabel = dict(bordercolor='white')
    )
    # Plot settings
    fig.update_layout(
        template='plotly_dark',
        # paper_bgcolor='rgba(0,0,0,0)',
        # plot_bgcolor='rgba(0,0,0,0)',
        height=600, width=1000,
        title=dict(text=f'Monthly {var} Climatology (\u00B0F) for {station}'.upper()+
                        '<br><sup>NOAA CO-OPS Site {} | {} - {}</sup>'.format(
                            stationid,
                            first_time.strftime('%m/%d/%Y'),
                            last_time.strftime('%m/%d/%Y')),
                   font=dict(size=20,
                              # family='PT Sans Narrow'
                         )),
        # yaxis = dict(title=f'{var} ({vk.units[var]})'.upper()),
        xaxis = dict(showgrid=False, showspikes=True,
                     dtick='M1', tickformat='%b %d'),
        hovermode='x unified',
        legend=dict(itemsizing='constant'),
        hoverlabel=dict(font_size=12,
                        # font_family="Rockwell"
                       )
    )
    if show:
        fig.show()
    else:
        return fig

# =============================================================================
# MAIN PROGRAM

def main():
    # Parse command line arguments
    args = parse_args()
    if not args.outdir:
        args.outdir = os.getcwd()

    # Download data
    data = Data(stationname=args.station, stationid=args.id, units=args.units,
                timezone=args.timezone, datum=args.datum, outdir=args.outdir,
                hr_threshold=args.hr,
                day_threshold=args.day,
                verbose=args.verbose)
    data.update_data()
    data.update_stats()

    # Plots
    vars = data.get_variables()
    plotDir = os.path.join(os.getcwd(), '_includes')
    if not os.path.exists(plotDir):
        os.makedir(plotDir)
    for var in vars:
        # Daily climatology
        dayfig = daily_climo(data.get_daily_stats_table(),
                    var=var,
                    station=data.get_station(),
                    stationid=data.get_stationid(),
                    first_time=data.filtered_data[var].dropna(axis=0).index.min(),
                    last_time=data.filtered_data[var].dropna(axis=0).index.max())
        fname = 'figure-{}-{}-daily.html'.format(
                    data.camel(data.get_station()).lower(),
                    var.lower().replace(' ', ''))
        pio.write_html(dayfig, file=os.path.join(plotDir, fname),
                       auto_open=True)

        # Monthly climatology
        monfig = monthly_climo(data.get_monthly_stats_table(),
                    var=var,
                    station=data.get_station(),
                    stationid=data.get_stationid(),
                    first_time=data.filtered_data[var].dropna(axis=0).index.min(),
                    last_time=data.filtered_data[var].dropna(axis=0).index.max())
        fname = 'figure-{}-{}-monthly.html'\
                    .format(data.camel(data.get_station()).lower(),
                            var.lower().replace(' ', ''))
        pio.write_html(monfig, file=os.path.join(plotDir, fname),
                       auto_open=True)

if __name__ == "__main__":
    main()