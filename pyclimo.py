from scipy.optimize import curve_fit
from noaa_coops import Station
from deepdiff import DeepDiff
import datetime as dt
import pandas as pd
import numpy as np
import calendar
import json
import os

class Data:
    def __init__(self, stationname, stationid, units='metric', timezone='gmt',
                datum='MHHW', outdir=None, hr_threshold=3, day_threshold=2,
                redownload=False, verbose=True):
        """Data class for downloading, formatting, and saving to file
        historical atmospheric (air temperature, barometric pressure, wind) and
        oceanographic (water temperature, water level) data from NOAA CO-OPS
        coastal tide stations.
        WARNING: This may take a while to initiate depending on the amount of
        data to be retrieved.
        
        Inputs:
            stationname: str, desired name of station. Does not need to be the
                CO-OPS name; it is used only for saving data to file.
            stationid: str, NOAA CO-OPS tide station number from which to
                retrieve data
            outdir: str, directory to save data to. Defaults to present working
                directory.
            units: str, either 'metric' or 'english', indicating the units to
                download data in. Defaults to 'metric'.
            timezone: str, one of either 'gmt' for Greenwich Mean Time, 'lst'
                for local standard time, or 'lst_ldt' for adjusted local
                standard/daylight savings time. Defaults to 'gmt'.
            datum: str, tidal datum for water level data. Options: 'STND',
                'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'MLLW', 'NAVD'. Defaults to
                'MHHW'.
            hr_threshold: int, maximum number of hours of data that can be
                missing in a given day in order for that day to be included in
                the historical record. Default is 3.
            day_threshold: int, maximum number of days of data that can be
                missing in a given month in order for that month to be included
                in the historical record. Default is 2.
            redownload: Bool, if True, historical data will be redownloaded and
                the class instance will be re-initiated. Defaults to False.
            verbose: Bool, print statuses to screen. Defaults to True.
        """
        
        self.name = stationname
        self.dirname = self.camel(stationname)
        self.id = stationid
        self.unit_system = units.lower()
        self.tz = timezone.lower()
        self.datum = datum.upper()
        self.hr_threshold = hr_threshold
        self.day_threshold = day_threshold
        self.verbose = verbose
        self.variables = []
        today = self._format_date(pd.to_datetime('today'))
        
        # Check for valid arguments
        self._valid_units(self.unit_system)
        self._valid_tz(self.tz)
        self._valid_datum(self.datum)
        
        # Set data directory, creating station subdirectory if needed
        if outdir:
            self.outdir = os.path.join(outdir, self.dirname)
        else:
            self.outdir = os.path.join(os.getcwd(), self.dirname)

        # =====================================================================
        # If 'redownload' argument is True OR if the directory station name
        # subdirectory does not exist within 'outdir', then create that
        # subdirectory and download historical data.
        if not os.path.exists(self.outdir) or redownload:
            if not os.path.exists(self.outdir):
                if self.verbose:
                    print('Creating new directory for this station.')
                os.makedirs(self.outdir)
        
            # Download all data and save to file
            self.download_data(start_date=None, end_date=None)
            outFile = os.path.join(self.outdir,
                                   'observational_data_record.csv.gz')
            self.data.to_csv(outFile, compression='infer')
            if self.verbose:
                print("Observational data written to file "\
                      f"'{outFile}'.")

            # Store units
            self.unit_options = dict({
                'Air Temperature': {'metric': 'C', 'english': 'F'},
                'Barometric Pressure': {'metric': 'mb', 'english': 'mb'},
                'Wind Speed': {'metric': 'm/s', 'english': 'kn'},
                'Wind Gust': {'metric': 'm/s', 'english': 'kn'},
                'Wind Direction': {'metric': 'deg', 'english': 'deg'},
                'Water Temperature': {'metric': 'C', 'english': 'F'},
                'Water Level': {'metric': 'm', 'english': 'ft'}
            })
            self.units =  {k:v[self.unit_system] \
                           for k,v in self.unit_options.items() \
                            if k in self.variables}
            
            # Save class variables
            meta = dict({
                'stationname': self.name,
                'stationid': self.id,
                'outdir': self.outdir,
                'unit_system': self.unit_system,
                'tz': self.tz,
                'datum': self.datum,
                'hr_threshold': self.hr_threshold,
                'day_threshold': self.day_threshold,
                'variables': self.variables,
                'units': self.units})
            with open(os.path.join(self.outdir, 'metadata.json'), 'w') as fp:
                json.dump(meta, fp) 
                    
            # Create and save statistics dictionaries
            self.filtered_data = {
                var: self._filter_data(self.data[var],
                                       hr_threshold=self.hr_threshold,
                                       day_threshold=self.day_threshold) \
                for var in self.variables}
            # Daily stats
            self.daily_stats_dict = self.daily_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-daily.json')
            with open(statsOutFile, 'w') as fp:
                json.dump(self.daily_stats_dict, fp)
            if self.verbose:
                print("Observational daily statistics written to "\
                      f"'{statsOutFile}'")
            # Monthly stats
            self.monthly_stats_dict = self.monthly_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.json')
            with open(statsOutFile, 'w') as fp:
                json.dump(self.monthly_stats_dict, fp)
            if self.verbose:
                print("Observational monthly statistics written to "
                      f"'{statsOutFile}'")            

        # =====================================================================
        # If historical data for this station already exists:
        else:
            # Load the metadata from file
            if self.verbose:
                print('Loading metadata from file')
            with open(os.path.join(self.outdir, 'metadata.json')) as m:
                meta = json.load(m)
            self._load_from_json(meta)
            
            # Load the historical data from file
            if self.verbose:
                print('Loading historical data from file')
            self.data = pd.read_csv(
                os.path.join(self.outdir, 'observational_data_record.csv.gz'),
                index_col=f'time_{self.tz}',
                parse_dates=True,
                compression='infer')
            
            # Load daily statistics from file
            if self.verbose:
                print('Loading daily statistics from file')
            with open(os.path.join(self.outdir, 'statistics-daily.json')) as s:
                self.daily_stats_dict = json.load(s)
            self.daily_stats_tables = self.stats_table(self.daily_stats_dict)

            # Load monthly statistics from file
            if self.verbose:
                print('Loading monthly statistics from file')
            with open(os.path.join(self.outdir, 'statistics-monthly.json')) as s:
                self.monthly_stats_dict = json.load(s)
            self.monthly_stats_tables = self.stats_table(self.monthly_stats_dict)

            # Clean and format
            self.filtered_data = {
                var: self._filter_data(self.data[var],
                                       hr_threshold=self.hr_threshold,
                                       day_threshold=self.day_threshold) \
                for var in self.variables}                
        if self.verbose:
            print('Done!')

    # =========================================================================
    def download_data(self, start_date=None, end_date=None):
        """Download data from NOAA CO-OPS"""
        if self.verbose:
            print('Downloading historic data')
        
        # NOAA CO-OPS API
        self.station = Station(id=self.id)

        # List of data variables to combine at the end
        datasets = []

        # If no 'end_date' is passed, download through current day
        if not end_date:
            end_date = self._format_date(pd.to_datetime('today'))

        # Air temperature
        if 'Air Temperature' in self.station.data_inventory:
            self.variables.append('Air Temperature')
            if not start_date:
                start_date = self._format_date(
                    self.station.data_inventory['Air Temperature']['start_date'])
            self._load_atemp(start_date=start_date, end_date=end_date)
            self.air_temp['atemp_flag'] = self.air_temp['atemp_flag'].str\
                                                .split(',', expand=True)\
                                                .astype(int)\
                                                .sum(axis=1)
            self.air_temp.loc[self.air_temp['atemp_flag']>0, 'atemp'] = np.nan
            datasets.append(self.air_temp['atemp'])

        # # Barometric pressure
        # if 'Barometric Pressure' in self.station.data_inventory:
        #     self.variables.append('Barometric Pressure')
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Barometric Pressure']['start_date'])
        #     self._load_atm_pres(start_date=start_date, end_date=end_date)
            # self.pressure['apres_flag'] = self.pressure['apres_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.pressure.loc[self.pressure['apres_flag'] > 0, 'apres'] = np.nan
        #     datasets.append(self.pressure['apres'])

        # # Wind
        # if 'Wind' in self.station.data_inventory:
        #     self.variables.extend(['Wind Speed', 'Wind Gust'])
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Wind']['start_date'])
        #     self._load_wind(start_date=start_date, end_date=end_date)
            # self.wind['windflag'] = self.wind['wind_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.wind.loc[self.wind['wind_flag'] > 0, ['windspeed', 'windgust']] = np.nan
        #     datasets.append(self.wind[['windspeed', 'windgust']])

        # Water temperature
        if 'Water Temperature' in self.station.data_inventory:
            self.variables.append('Water Temperature')
            if not start_date:
                start_date = self._format_date(
                    self.station.data_inventory['Water Temperature']['start_date'])
            self._load_water_temp(start_date=start_date, end_date=end_date)
            self.water_temp['wtemp_flag'] = self.water_temp['wtemp_flag'].str\
                                                    .split(',', expand=True)\
                                                    .astype(int)\
                                                    .sum(axis=1)
            self.water_temp.loc[self.water_temp['wtemp_flag']>0, 'wtemp'] = np.nan
            datasets.append(self.water_temp['wtemp'])

        # # Water level (tides)
        # if 'Verified 6-Minute Water Level' in self.station.data_inventory:
        #     self.variables.append('Water Level')
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Verified 6-Minute Water Level']['start_date'])
        #     self._load_water_level(start_date=start_date, end_date=end_date)
            # self.water_levels['wlevel_flag'] = self.water_levels['wlevel_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.water_levels.loc[self.water_levels['wlevel_flag'] > 0, 'wlevel'] = np.nan
        #     datasets.append(self.water_levels['wlevel'])

        # Merge into single dataframe
        if self.verbose:
            print('Compiling data')
        self.data = pd.concat(datasets, axis=1)
        self.data.index.name = f'time_{self.tz}'
        self.data.columns = [i for i in self.variables]

    def update_data(self, start_date=None, end_date=None):
        """Download data from NOAA CO-OPS"""
        if self.verbose:
            print('Downloading latest data')

        # NOAA CO-OPS API
        self.station = Station(id=self.id)

        # List of data variables to combine at the end
        datasets = []
        
        # If no 'start_date' is passed, pick up from the last observation time
        if not start_date:
            start_date = self._format_date(self.data.index.max())
            
        # If no 'end_date' is passed, download through end of current date
        if not end_date:
            end_date = self._format_date(pd.to_datetime('today') + pd.Timedelta(days=1))
        
        # Air temperature
        if 'Air Temperature' in self.variables:
            self._load_atemp(start_date=start_date, end_date=end_date)
            datasets.append(self.air_temp['atemp'])

        # Barometric pressure
        if 'Barometric Pressure' in self.variables:
            self._load_atm_pres(start_date=start_date, end_date=end_date)
            datasets.append(self.pressure['apres'])

        # Wind
        if 'Wind Speed' in self.variables:
            self._load_wind(start_date=start_date, end_date=end_date)
            datasets.append(self.wind[['windspeed', 'windgust']])

        # Water temperature
        if 'Water Temperature' in self.variables:
            self._load_water_temp(start_date=start_date, end_date=end_date)
            datasets.append(self.water_temp['wtemp'])

        # Water level (tides)
        if 'Verified 6-Minute Water Level' in self.variables:
            self._load_water_level(start_date=start_date, end_date=end_date)
            datasets.append(self.water_levels['wlevel'])

        # Merge into single dataframe
        data = pd.concat(datasets, axis=1)
        if sum(~data.index.isin(self.data.index)) == 0:
            print('No new data available.')
        else:
            data.index.name = f'time_{self.tz}'
            data.columns = [i for i in self.variables]
            data = pd.concat([self.data,
                              data[data.index.isin(self.data.index) == False]],
                             axis=0)
            self.data = data
            self.filtered_data = {var:self._filter_data(self.data[var],
                                                        hr_threshold=self.hr_threshold,
                                                        day_threshold=self.day_threshold).sort_values(['YearDay']) \
                                 for var in self.variables}
            self.data.to_csv(os.path.join(self.outdir, 'observational_data_record.csv.gz'),
                             compression='infer')
            if self.verbose:
                print("Updated observational data written to file "\
                      f"'{os.path.join(self.outdir, 'observational_data_record.csv')}'.")
                print("Done! (Don't forget to run Data.update_stats() to update statistics.)")
    
    def update_stats(self):    
        """Calculate new statistics and update if any changes"""
        # Daily stats
        _new_daily_stats = self.daily_stats()
        if self._ordered(_new_daily_stats) != self._ordered(self.daily_stats_dict):
            if self.verbose:
                print('Daily stats dicts differ. Updating and saving to file.\n')
                print('*'*10)
                print('NEW RECORD')
                self._compare(self.daily_stats_dict, _new_daily_stats)
                print('*'*10)
            self.daily_stats_dict = _new_daily_stats
            # Write to file
            statsOutFile = os.path.join(self.outdir, 'statistics-daily.json')
            with open(statsOutFile, 'w') as fp:
                json.dump(self.daily_stats_dict, fp)
            if self.verbose:
                print(f"\nUpdated daily observational statistics written to '{statsOutFile}'")
        else:
            if self.verbose:
                print("No new daily records set.")

        # Monthly stats
        _new_monthly_stats = self.monthly_stats()
        if self._ordered(_new_monthly_stats) != self._ordered(self.monthly_stats_dict):
            if self.verbose:
                print('Monthly stats dicts differ. Updating and saving to file.\n')
                print('*'*10)
                print('NEW RECORD')
                self._compare(self.monthly_stats_dict, _new_monthly_stats)
                print('*'*10)
            self.monthly_stats_dict = _new_monthly_stats
            # Write to file
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.json')
            with open(statsOutFile, 'w') as fp:
                json.dump(self.monthly_stats_dict, fp)
            if self.verbose:
                print(f"\nUpdated daily observational statistics written to '{statsOutFile}'")
        else:
            if self.verbose:
                print("No new monthly records set.")

    def _ordered(self, obj):
        if isinstance(obj, dict):
            return sorted((k, self._ordered(v)) for k, v in obj.items())
        if isinstance(obj, list):
            return sorted(self._ordered(x) for x in obj)
        else:
            return obj

    def _format_date(self, datestr):
        dtdt = pd.to_datetime(datestr)
        return dt.datetime.strftime(dtdt, '%Y%m%d')
    
    def camel(self, text):
        """Convert to camel case"""
        s = text.replace(',','').replace("-", " ").replace("_", " ")
        s = s.split()
        if len(text) == 0:
            return text
        return s[0].lower() + ''.join(i.capitalize() for i in s[1:])

    def _valid_units(self, unit):
        valid = {'metric', 'english'}
        if unit.lower() not in valid:
            raise ValueError("units: units must be one of %r." % valid)
    
    def _valid_tz(self, tz):
        valid = {'gmt', 'lst', 'lst_ldt'}
        if tz.lower() not in valid:
            raise ValueError("timezone: timezone must be one of %r." % valid)

    def _valid_datum(self, datum):
        valid = {'STND', 'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'MLLW', 'NAVD'}
        if datum.upper() not in valid:
            raise ValueError("datum: datum must be one of %r." % valid)

    def _load_from_json(self, blob):
        for k, v in blob.items():
            setattr(self, k, v)
    
    def get_data(self):
        return self.data
        
    def _load_atemp(self, start_date, end_date):
        """Download air temperature data from NOAA CO-OPS from 'start_date'
        through current day.
        """
        if self.verbose:
            print('Retrieving air temperature data')
        self.air_temp = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='air_temperature',
            units=self.unit_system,
            time_zone=self.tz)
        self.air_temp.columns = ['atemp', 'atemp_flag']
    
    def _load_wind(self, start_date, end_date):
        """Download wind data from NOAA CO-OPS from 'start_date' through
        current day.
        """
        if self.verbose:
            print('Retrieving wind data')
        self.wind = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='wind',
            units=self.unit_system,
            time_zone=self.tz)
        self.wind.columns = ['windspeed', 'winddir_deg', 'winddir',
                             'windgust', 'wind_flag']
    
    def _load_atm_pres(self, start_date, end_date):
        """Download barometric pressure data from NOAA CO-OPS from 'start_date'
        through current day.
        """
        if self.verbose:
            print('Retrieving barometric pressure data')
        self.pressure = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='air_pressure',
            units=self.unit_system,
            time_zone=self.tz)
        self.pressure.columns = ['apres', 'apres_flag']
    
    def _load_water_temp(self, start_date, end_date):
        """Download water temperature data from NOAA CO-OPS from 'start_date'
       through current day.
        """
        if self.verbose:
            print('Retrieving water temperature data')
        self.water_temp = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='water_temperature',
            units=self.unit_system,
            time_zone=self.tz)
        self.water_temp.columns = ['wtemp', 'wtemp_flag']

    def _load_water_level(self, start_date, end_date):
        """Download water level tide data from NOAA CO-OPS from 'start_date'
        through current day.
        """
        if self.verbose:
            print('Retrieving water level tide data')
        self.water_levels = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='water_level',
            datum=self.datum,
            units=self.unit_system,
            time_zone=self.tz)
        self.water_levels.columns = ['wlevel', 's', 'wlevel_flag', 'wlevel_qc']

    def _DOY(self, df):
        """Calculate year day out of 366"""
        import calendar
        # Day of year as integer
        df['YearDay'] = df.index.day_of_year.astype(int)
        # Years that are NOT leap years
        leapInd = [not calendar.isleap(i) for i in df.index.year]
        mask = (leapInd) & (df['Month'] > 2)
        # Advance by one day everything after February 28 
        df.loc[mask, 'YearDay'] += 1
        # Day of year for plotting
        # df['DOY'] = pd.to_datetime(df['YearDay'].astype(str), format="%j")
        df['DOY'] = pd.to_datetime(df.index.day_of_year.astype(str), format="%j")
        return df

    def _set_dayflag(self, data, threshold_hrs=3):
        """Set flag indicating whether each day contains at least
        (24 - 'threshold_hours') hours of data: True if yes, False if not.
        """
        def _flag(group, threshold=threshold_hrs):
            # gap = sum(group.index.to_series().diff() > pd.Timedelta(hours=1))
            gap = group.resample('1h').mean(numeric_only=True).isna().sum()
            return gap <= threshold

        newColName = data.name + '_DayFlag'
        data = pd.DataFrame(data[:])
        data[newColName] = data.groupby(pd.Grouper(freq='D')).agg(
            lambda x: _flag(group=x, threshold=threshold_hrs))
        data[newColName] = data.resample('D')[newColName].ffill()
        data = data.astype({newColName: 'bool'})
        return data
    
    def _set_monthflag(self, data, threshold_days=2):
        """Set flag indicating whether each month contains at least
        (n -'threshold_days') days of data, where n is the number of days in
        the given month: True if yes, False if not.
        """
        def _flag(group, threshold=threshold_days):
            try:
                # gap = sum(group.index.to_series().diff() > pd.Timedelta(days=1))
                # flag = gap <= threshold
                daysWithData = group.resample('1D').mean(numeric_only=True).dropna().size
                daysNoData =  pd.Period(group.index[0].strftime(format='%Y-%m-%d')).days_in_month - daysWithData
                flag = daysNoData <= threshold
                return pd.DataFrame(pd.Series(flag, index=group.index))
            except IndexError:
                return None

        origColName = data.name
        newColName = data.name + '_MonthFlag'
        data = pd.DataFrame(data[:])
        data['Year'] = data.index.year
        data['Month'] = data.index.month
        data['Day'] = data.index.day
        data[newColName] = data.groupby(['Year', 'Month'], group_keys=False)\
                               .apply(lambda x: _flag(x, threshold_days))
        return data

    def _filter_data(self, series, hr_threshold=3, day_threshold=2):
        """Removes days with more than 'hr_threshold' of missing data and
        months with more than 'day_threshold' days of missing data.
        """
        # Variable to process
        var = series.name

        # Remove days with >hr_threshold hours of missing data
        df_dayFlagged = self._set_dayflag(series[:],
                                          threshold_hrs=hr_threshold)
        dfss = df_dayFlagged.loc[df_dayFlagged[f'{var}_DayFlag'],:]

        # Remove months with >day_threshold days of missing data
        df_monFlagged = self._set_monthflag(dfss[var],
                                            threshold_days=day_threshold)
        dfss = df_monFlagged.loc[df_monFlagged[f'{var}_MonthFlag'],:].copy()
        
        # Add Year Day
        dfss = self._DOY(dfss)

        # Drop month flag
        dfss.drop(f'{var}_MonthFlag', axis=1, inplace=True)
        return dfss.sort_index()
            
    def daily_highs(self, var, decimals=1):
        """Daily highs for variable 'var'."""
        return self.filtered_data[var].groupby(pd.Grouper(freq='D'))\
                                      .max(numeric_only=True)
    
    def daily_lows(self, var, decimals=1):
        """Daily lows for variable 'var'."""
        return self.filtered_data[var].groupby(pd.Grouper(freq='D'))\
                                      .min(numeric_only=True)

    def daily_avgs(self, var, decimals=1, true_average=False):
        """Daily averages by calendar day for variable 'var' rounded to
        'decimals'. If 'true_average' is True, all measurements from each
        24-hour day will be used to calculate the average. Otherwise, only the
        maximum and minimum observations are used. Defaults to False.
        """
        if true_average:
            return self.filtered_data[var].groupby(pd.Grouper(freq='D'))\
                                          .mean(numeric_only=True).round(decimals)
        else:
            dailyHighs = self.daily_highs(var=var, decimals=decimals)
            dailyLows = self.daily_lows(var=var, decimals=decimals)
            return ((dailyHighs + dailyLows)/2).round(decimals)

    def daily_avg(self, var, decimals=1):
        """Daily averages for variable 'var' rounded to 'decimals'."""
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        dailyAvg = dailyAvgs.groupby('YearDay').mean(numeric_only=True)\
                                               .round(decimals)[var]
        dailyAvg.index = dailyAvg.index.astype(int)
        return dailyAvg
    
    def monthly_avg(self, var, decimals=1):
        """Monthly averages for variable 'var' rounded to 'decimals'."""
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        monthlyMeans = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                                .mean(numeric_only=True).round(decimals)
        monthlyAvg = monthlyMeans.groupby('Month')\
                                 .mean(numeric_only=True).round(decimals)[var]
        monthlyAvg.index = monthlyAvg.index.astype(int)
        return monthlyAvg

    def monthly_highs(self, var, decimals=1):
        """Monthly highs for variable 'var'."""
        dailyAvgs = self.daily_avgs(var=var)
        return dailyAvgs.groupby(pd.Grouper(freq='M')).max(numeric_only=True)
      
    def monthly_lows(self, var, decimals=1):
        """Monthly lows for variable 'var'."""
        dailyAvgs = self.daily_avgs(var=var)
        return dailyAvgs.groupby(pd.Grouper(freq='M')).min(numeric_only=True)

    def record_high_daily_avg(self, var, decimals=1):
        """Record high daily averages for variable 'var' rounded to
        'decimals'.
        """
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        recordHighDailyAvg = dailyAvgs.groupby('YearDay')\
                                      .max(numeric_only=True)\
                                      .round(decimals)[var]
        recordHighDailyAvg.index = recordHighDailyAvg.index.astype(int)
        recordHighDailyAvgYear = dailyAvgs.groupby('YearDay')\
                                          .idxmax(numeric_only=True)[var]\
                                          .dt.year.astype(int)
        recordHighDailyAvgYear.index = recordHighDailyAvgYear.index.astype(int)
        return (recordHighDailyAvg, recordHighDailyAvgYear)
    
    def record_high_monthly_avg(self, var, decimals=1):
        """Record high monthly averages for variable 'var' rounded to
        'decimals'.
        """
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True).round(decimals)
        recordHighMonthlyAvg = monthlyAvgs.groupby('Month')\
                                          .max(numeric_only=True)[var]
        recordHighMonthlyAvg.index = recordHighMonthlyAvg.index.astype(int)
        recordHighMonthlyAvgYear = monthlyAvgs.groupby('Month')\
                                              .idxmax(numeric_only=True)[var]\
                                              .dt.year.astype(int)
        recordHighMonthlyAvgYear.index = recordHighMonthlyAvgYear.index.astype(int)
        return (recordHighMonthlyAvg, recordHighMonthlyAvgYear)
    
    def record_low_daily_avg(self, var, decimals=1):
        """Record low daily averages for variable 'var' rounded to
        'decimals'.
        """
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        recordLowDailyAvg = dailyAvgs.groupby('YearDay')\
                                     .min(numeric_only=True).round(decimals)[var]
        recordLowDailyAvg.index = recordLowDailyAvg.index.astype(int)
        recordLowDailyAvgYear = dailyAvgs.groupby('YearDay')\
                                         .idxmin(numeric_only=True)[var]\
                                         .dt.year.astype(int)
        recordLowDailyAvgYear.index = recordLowDailyAvg.index.astype(int)
        return (recordLowDailyAvg, recordLowDailyAvgYear)

    def record_low_monthly_avg(self, var, decimals=1):
        """Record low monthly averages for variable 'var' rounded to
        'decimals'.
        """
        dailyAvgs = self.daily_avgs(var=var, decimals=decimals)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True).round(decimals)
        recordLowMonthlyAvg = monthlyAvgs.groupby('Month')\
                                         .min(numeric_only=True)[var]
        recordLowMonthlyAvg.index = recordLowMonthlyAvg.index.astype(int)
        recordLowMonthlyAvgYear = monthlyAvgs.groupby('Month')\
                                             .idxmin(numeric_only=True)[var]\
                                             .dt.year.astype(int)
        recordLowMonthlyAvgYear.index = recordLowMonthlyAvgYear.index.astype(int)
        return (recordLowMonthlyAvg, recordLowMonthlyAvgYear)

    def daily_avg_high(self, var, decimals=1):
        """Average daily highs for variable 'var' rounded to 'decimals'."""
        dailyHighs = self.daily_highs(var=var, decimals=decimals)
        return dailyHighs.groupby('YearDay')\
                         .mean(numeric_only=True).round(decimals)[var]
    
    def monthly_avg_high(self, var, decimals=1):
        """Average monthly highs for variable 'var' rounded to 'decimals'."""
        monthlyHighs = self.monthly_highs(var=var, decimals=decimals)
        return monthlyHighs.groupby('Month')\
                           .mean(numeric_only=True).round(decimals)[var]
    
    def daily_lowest_high(self, var, decimals=1):
        """Lowest daily highs for variable 'var' rounded to 'decimals'."""
        dailyHighs = self.daily_highs(var=var, decimals=decimals)
        lowestHigh = dailyHighs.groupby('YearDay')\
                               .min(numeric_only=True).round(decimals)[var]
        lowestHighYear = dailyHighs.groupby('YearDay')\
                                   .idxmin(numeric_only=True)[var]\
                                   .dt.year.astype(int)
        return (lowestHigh, lowestHighYear)
    
    def monthly_lowest_high(self, var, decimals=1):
        """Lowest monthly highs for variable 'var' rounded to 'decimals'."""
        monthlyHighs = self.monthly_highs(var=var, decimals=decimals)
        lowestHigh = monthlyHighs.groupby('Month').min(numeric_only=True)[var]
        lowestHighYear = monthlyHighs.groupby('Month')\
                                     .idxmin(numeric_only=True)[var]\
                                     .dt.year.astype(int)
        return (lowestHigh, lowestHighYear)
        
    def daily_record_high(self, var, decimals=1):
        """Record daily highs for variable 'var' rounded to 'decimal'."""
        dailyHighs = self.daily_highs(var=var, decimals=decimals)
        recordHigh = dailyHighs.groupby('YearDay')\
                               .max(numeric_only=True).round(decimals)[var]
        recordHighYear = dailyHighs.groupby('YearDay')\
                                   .idxmax(numeric_only=True)[var]\
                                   .dt.year.astype(int)
        return (recordHigh, recordHighYear)

    def monthly_record_high(self, var, decimals=1):
        """Record monthly highs for variable 'var' rounded to 'decimals'."""
        monthlyHighs = self.monthly_highs(var=var, decimals=decimals)
        recordHigh = monthlyHighs.groupby('Month').max(numeric_only=True)[var]
        recordHighYear = monthlyHighs.groupby('Month')\
                                     .idxmax(numeric_only=True)[var]\
                                     .dt.year.astype(int)
        return (recordHigh, recordHighYear)

    def daily_avg_low(self, var, decimals=1):
        """Average daily lows for variable 'var' rounded to 'decimals'."""
        dailyLows = self.daily_lows(var=var, decimals=decimals)
        return dailyLows.groupby('YearDay')\
                        .mean(numeric_only=True).round(decimals)[var]
    
    def monthly_avg_low(self, var, decimals=1):
        """Average monthly lows for variable 'var' rounded to 'decimals'."""
        monthlyLows = self.monthly_lows(var=var, decimals=decimals)
        return monthlyLows.groupby('Month')\
                          .mean(numeric_only=True).round(decimals)[var]
    
    def daily_highest_low(self, var, decimals=1):
        """Highest daily lows for variable 'var' rounded to 'decimals'."""
        dailyLows = self.daily_lows(var=var, decimals=decimals)
        highestLow =  dailyLows.groupby('YearDay')\
                               .max(numeric_only=True).round(decimals)[var]
        highestLowYear =  dailyLows.groupby('YearDay')\
                                   .idxmax(numeric_only=True)[var]\
                                   .dt.year.astype(int)
        return (highestLow, highestLowYear)
    
    def monthly_highest_low(self, var, decimals=1):
        """Highest monthly lows for variable 'var' rounded to 'decimals'."""
        monthlyLows = self.monthly_lows(var=var, decimals=decimals)
        highestLow = monthlyLows.groupby('Month').max(numeric_only=True)[var]
        highestLowYear = monthlyLows.groupby('Month')\
                                    .idxmax(numeric_only=True)[var]\
                                    .dt.year.astype(int)
        return (highestLow, highestLowYear)
    
    def daily_record_low(self, var, decimals=1):
        """Record daily lows for variable 'var' rounded to 'decimals'."""
        dailyLows = self.daily_lows(var=var, decimals=decimals)
        recordLow = dailyLows.groupby('YearDay')\
                             .min(numeric_only=True).round(decimals)[var]
        recordLowYear = dailyLows.groupby('YearDay')\
                                 .idxmin(numeric_only=True)[var]\
                                 .dt.year.astype(int)
        return (recordLow, recordLowYear)
    
    def monthly_record_low(self, var, decimals=1):
        """Record monthly lows for variable 'var' rounded to 'decimals'."""
        monthlyLows = self.monthly_lows(var=var, decimals=decimals)
        recordLow = monthlyLows.groupby('Month').min(numeric_only=True)[var]
        recordLowYear = monthlyLows.groupby('Month')\
                                   .idxmin(numeric_only=True)[var]\
                                   .dt.year.astype(int)
        return (recordLow, recordLowYear)
    
    def number_of_years_byday(self, var):
        """Number of years in the historical data record for variable 'var' by
        day of year.
        """
        return self.filtered_data[var].groupby('YearDay').apply(
                    lambda x: len(x['Year'].unique()))

    def number_of_years_bymon(self, var):
        """Number of years in the historical data record for variable 'var' by
        month.
        """
        return self.filtered_data[var].groupby('Month').apply(
                    lambda x: len(x['Year'].unique()))

    def generate_yeardays(self):
        return pd.date_range(start='2020-01-01',
                             end='2020-12-31',
                             freq='1D').strftime('%d-%b')
    
    def daily_stats(self):
        """Create daily statistics dictionary for all science variables."""
        # Use a leap year to generate a date list
        YearDay = pd.date_range(
            start='2020-01-01',
            end='2020-12-31',
            freq='1D').strftime('%d-%b')
        # New dictionary to fill
        stats_dict = {k:None for k in self.variables}
        for var in stats_dict.keys():
            if self.verbose:
                print(f'Calculating daily {var.lower()} stats')
            decimals = 2 if var=='Water Level' else 1
            
            # Extract data when multiple outputs
            recordHighDailyAvg, recordHighDailyAvgYr = \
                self.record_high_daily_avg(var=var, decimals=decimals)
            recordLowDailyAvg, recordLowDailyAvgYr = \
                self.record_low_daily_avg(var=var, decimals=decimals)
            lowestHigh, lowestHighYr = \
                self.daily_lowest_high(var=var, decimals=decimals)
            recordHigh, recordHighYr = \
                self.daily_record_high(var=var, decimals=decimals)
            highestLow, highestLowYr = \
                self.daily_highest_low(var=var, decimals=decimals)
            recordLow, recordLowYr = \
                self.daily_record_low(var=var, decimals=decimals)
            
            stats_dict[var] = dict({
                # 'DailyHighs': {d: {'obs': o} \
                #      for d,o in zip(self.daily_highs(var=var, decimals=decimals).index.strftime('%Y-%m-%d').to_list(),
                #                     self.daily_highs(var=var, decimals=decimals)[var].to_list())},
                # 'DailyLows': {d: {'obs': o} \
                #      for d,o in zip(self.daily_lows(var=var, decimals=decimals).index.strftime('%Y-%m-%d').to_list(),
                #                     self.daily_lows(var=var, decimals=decimals)[var].to_list())},
                 'Daily Average': {d: {'obs': o} \
                     for d,o in zip(YearDay,
                                    self.daily_avg(
                                        var=var,
                                        decimals=decimals).to_list())},
                'Record High Daily Average': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      recordHighDailyAvg.to_list(),
                                      recordHighDailyAvgYr.to_list())},
                'Record Low Daily Average': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      recordLowDailyAvg.to_list(),
                                      recordLowDailyAvgYr.to_list())},
                'Average High': {d: {'obs': o} \
                     for d,o in zip(YearDay,
                                    self.daily_avg_high(
                                        var=var,
                                        decimals=decimals).to_list())},
                'Lowest High': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      lowestHigh.to_list(),
                                      lowestHighYr.to_list())},
                'Record High': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      recordHigh.to_list(),
                                      recordHighYr.to_list())},
                'Average Low': {d: {'obs': o} \
                     for d,o in zip(YearDay,
                                      self.daily_avg_low(
                                          var=var,
                                          decimals=decimals).to_list())},
                'Highest Low': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      highestLow.to_list(),
                                      highestLowYr.to_list())},
                'Record Low': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(YearDay,
                                      recordLow.to_list(),
                                      recordLowYr.to_list())},
                'Number of Years': {d: {'obs': o} \
                     for d,o in zip(YearDay,
                                      self.number_of_years_byday(var=var).to_list())}
            })
        
        # Create table version
        self.daily_stats_tables = self.stats_table(stats_dict)
        return stats_dict    

    def monthly_stats(self):
        """Create monthly statistics dictionary for all science variables."""
        # Use a leap year to generate a date list
        months = pd.date_range(
            start='2020-01-01',
            end='2020-12-31',
            freq='1M').strftime('%b')
        # New dictionary to fill
        stats_dict = {k:None for k in self.variables}
        for var in stats_dict.keys():
            if self.verbose:
                print(f'Calculating monthly {var.lower()} stats')
            decimals = 2 if var=='Water Level' else 1
            
            # Extract data when multiple outputs
            recordHighMonthlyAvg, recordHighMonthlyAvgYr = \
                self.record_high_monthly_avg(var=var, decimals=decimals)
            recordLowMonthlyAvg, recordLowMonthlyAvgYr = \
                self.record_low_monthly_avg(var=var, decimals=decimals)
            lowestHigh, lowestHighYr = \
                self.monthly_lowest_high(var=var, decimals=decimals)
            recordHigh, recordHighYr = \
                self.monthly_record_high(var=var, decimals=decimals)
            highestLow, highestLowYr = \
                self.monthly_highest_low(var=var, decimals=decimals)
            recordLow, recordLowYr = \
                self.monthly_record_low(var=var, decimals=decimals)
            
            stats_dict[var] = dict({
                # 'MonthlyHighs': {d: {'obs': o} \
                #      for d,o in zip(self.monthly_highs(var=var, decimals=decimals).index.strftime('%Y-%m-%d').to_list(),
                #                     self.monthly_highs(var=var, decimals=decimals)[var].to_list())},
                # 'MonthlyLows': {d: {'obs': o} \
                #      for d,o in zip(self.daily_lows(var=var, decimals=decimals).index.strftime('%Y-%m-%d').to_list(),
                #                     self.daily_lows(var=var, decimals=decimals)[var].to_list())},
                 'Monthly Average': {d: {'obs': o} \
                     for d,o in zip(months,
                                    self.monthly_avg(
                                        var=var,
                                        decimals=decimals).to_list())},
                'Record High Monthly Average': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      recordHighMonthlyAvg.to_list(),
                                      recordHighMonthlyAvgYr.to_list())},
                'Record Low Monthly Average': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      recordLowMonthlyAvg.to_list(),
                                      recordLowMonthlyAvgYr.to_list())},
                'Average High': {d: {'obs': o} \
                     for d,o in zip(months,
                                    self.monthly_avg_high(
                                        var=var,
                                        decimals=decimals).to_list())},
                'Lowest High': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      lowestHigh.to_list(),
                                      lowestHighYr.to_list())},
                'Record High': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      recordHigh.to_list(),
                                      recordHighYr.to_list())},
                'Average Low': {d: {'obs': o} \
                     for d,o in zip(months,
                                      self.monthly_avg_low(
                                          var=var,
                                          decimals=decimals).to_list())},
                'Highest Low': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      highestLow.to_list(),
                                      highestLowYr.to_list())},
                'Record Low': {d: {'obs': o, 'year':y} \
                     for d,o,y in zip(months,
                                      recordLow.to_list(),
                                      recordLowYr.to_list())},
                'Number of Years': {d: {'obs': o} \
                     for d,o in zip(months,
                                    self.number_of_years_bymon(var=var).to_list())}
            })
        
        # Create table version
        self.monthly_stats_tables = self.stats_table(stats_dict)
        return stats_dict    
    
    def stats_table(self, stats_dict):
        """Convert statistics dictionary to DataFrame for easier viewing"""
        _tables = {k:None for k in self.variables}
        for var in self.variables:
            out = pd.DataFrame.from_dict(stats_dict[var])
            for col in out.columns:
                new = pd.json_normalize(out[col])
                if len(new.columns) == 2:
                    new.columns = [col, col+' Year']
                else:
                    new.columns = [col]
                new.index = out.index
                out.drop(col, axis=1, inplace=True)
                out = pd.concat((out, new), axis=1)
            _tables[var] = out
        return _tables

    def _report(self, record, variable, ondate, values, years):
        """Print new records"""
        oldRecord = values['old_value']
        oldYear = years['old_value']
        newRecord = values['new_value']        
        units = self.units[variable]
        print(f"{record.capitalize()} {variable.lower()} set {ondate}:\n\t"\
              f"{newRecord} {units} (previously {oldRecord} {units} in {oldYear})")
        
    def _compare(self, d1, d2):
        """Compare dictionaries excluding daily highs, lows, and averages,
        since these will always change with updated data"""
        exclude = [f"root['{v}']['{e}']" \
                   for e in ['Daily Average', 'Monthly Average',
                             'Average High', 'Average Low',
                             'Number of Years'] \
                   for v in self.variables]
        deltas = DeepDiff(d1, d2, exclude_paths=exclude)
        # Print any new records
        if deltas:
            keylist = list(deltas['values_changed'].keys())
            for i in np.arange(0, len(keylist), 2):
                variable, record, ondate, _ = keylist[i][6:].split("']['")
                self._report(record, variable, ondate,
                             values=deltas['values_changed'][keylist[i]],
                             years=deltas['values_changed'][keylist[i+1]])
    
    def get_daily_stats(self, var=None):
        """Return the daily statistics dictionary"""
        try:
            return self.daily_stats_dict[var] if var else self.daily_stats_dict
        except AttributeError:
            raise AttributeError('Instance of Data has no daily stats yet. '\
                                 'Run Data.stats() to calculate stats and '\
                                 'try again.')
    
    def get_monthly_stats(self, var=None):
        """Return the monthly statistics dictionary"""
        try:
            return self.monthly_stats_dict[var] \
                       if var else self.monthly_stats_dict
        except AttributeError:
            raise AttributeError('Instance of Data has no monthly stats yet. '\
                                 'Run Data.stats() to calculate stats and '\
                                 'try again.')

    def get_daily_stats_table(self, var=None):
        """Return the daily statistics table"""
        try:
            return self.daily_stats_tables[var] \
                       if var else self.daily_stats_tables
        except AttributeError:
            raise AttributeError('Instance of Data has no daily stats table '\
                                 'yet. Run Data.daily_stats() to calculate '\
                                 'stats and try again.')
            
    def get_monthly_stats_table(self, var=None):
        """Return the monthly statistics table"""
        try:
            return self.monthly_stats_tables[var] \
                       if var else self.monthly_stats_tables
        except AttributeError:
            raise AttributeError('Instance of Data has no monthly stats '\
                                 'table yet. Run Data.monthly_stats() to '\
                                 'calculate stats and try again.')

    def _skip_keys(self, d, keys):
        return {x: d[x] for x in d if x not in keys}

    def set_station(self, station):
        self.name = station
        
    def get_station(self):
        return self.name
    
    def get_stationid(self):
        return self.id
    
    def get_variables(self):
        return self.variables
    
    def __str__(self):
        return("Oceanic and atmospheric observations for station "\
               f"'{self.name}' (station ID {self.id}):\n"\
               f"{self.station.data_inventory}")
    
    def __repr__(self):
        return(f"{type(self).__name__}("\
               f"stationname='{self.name}', stationid='{self.id}', "\
               f"outdir='{self.outdir}', timezone='{self.tz}', "\
               f"units='{self.units}', datum='{self.datum}', "\
               f"hr_threshold='{self.hr_threshold}', "\
               f"day_threshold='{self.day_threshold}')")