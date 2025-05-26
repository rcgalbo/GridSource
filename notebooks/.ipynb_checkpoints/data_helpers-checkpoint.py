"""
Data exploration helper functions for GridSource research

This module provides easy-to-use functions for exploring API data sources
and understanding the structure of data in the energy banking pipeline.
"""

import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import warnings
import os
from pathlib import Path


class APIExplorer:
    """Helper class for exploring external APIs used in the pipeline"""
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize API explorer with optional API keys
        
        Args:
            api_keys: Dictionary with keys like 'eia_api_key', 'fred_api_key'
        """
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GridSource Research Tool'
        })
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")

    def explore_eia_api(self, api_key: Optional[str] = None, 
                       days: int = 7, detailed: bool = True) -> pd.DataFrame:
        """
        Explore EIA electricity generation API
        
        Args:
            api_key: EIA API key (uses self.api_keys if not provided)
            days: Number of days of data to fetch
            detailed: Whether to show detailed information
            
        Returns:
            DataFrame with EIA data
        """
        api_key = api_key or self.api_keys.get('eia_api_key')
        if not api_key:
            print("⚠️  No EIA API key provided. Get one at: https://www.eia.gov/opendata/register.php")
            return self._create_sample_eia_data(days)
        
        print(f"🔍 Exploring EIA Electricity API (last {days} days)...")
        
        base_url = "https://api.eia.gov/v2/electricity/rto/daily-region-data/data/"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            'api_key': api_key,
            'frequency': 'daily',
            'data[0]': 'value',
            'facets[respondent][]': 'CAL',  # California
            'facets[fueltype][]': ['NG', 'SUN', 'WAT', 'WND'],  # Multiple fuel types
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d'),
            'sort[0][column]': 'period',
            'sort[0][direction]': 'desc',
            'length': 100000  # Get ALL available data (API max)
        }
        
        try:
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Transform to DataFrame
            df = pd.DataFrame(data['response']['data'])
            df['date'] = pd.to_datetime(df['period'])
            df['generation_mwh'] = pd.to_numeric(df['value'], errors='coerce')
            df['fuel_type'] = df['fueltype']
            
            if detailed:
                self._show_eia_details(df, data)
                
            return df[['date', 'fuel_type', 'generation_mwh', 'respondent']].dropna()
            
        except Exception as e:
            print(f"❌ Error fetching EIA data: {str(e)}")
            return self._create_sample_eia_data(days)

    def explore_weather_api(self, detailed: bool = True) -> pd.DataFrame:
        """
        Explore NOAA weather API
        
        Args:
            detailed: Whether to show detailed information
            
        Returns:
            DataFrame with weather data
        """
        print("🔍 Exploring NOAA Weather API...")
        
        url = "https://api.weather.gov/gridpoints/MTR/90,112/forecast"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Extract forecast periods
            periods = data['properties']['periods']
            
            weather_data = []
            for period in periods:
                weather_data.append({
                    'date': period['startTime'][:10],
                    'period_name': period['name'],
                    'temperature_f': period.get('temperature', 0),
                    'wind_speed': period.get('windSpeed', 'Unknown'),
                    'wind_direction': period.get('windDirection', 'Unknown'),
                    'forecast': period.get('shortForecast', ''),
                    'detailed_forecast': period.get('detailedForecast', '')
                })
            
            df = pd.DataFrame(weather_data)
            
            if detailed:
                self._show_weather_details(df, data)
                
            return df
            
        except Exception as e:
            print(f"❌ Error fetching weather data: {str(e)}")
            return self._create_sample_weather_data()

    def get_historical_weather_data(self, start_date: str, end_date: str, 
                                   location: str = "San Francisco, CA",
                                   detailed: bool = True) -> pd.DataFrame:
        """
        Get historical weather data using multiple sources
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            location: Location for weather data
            detailed: Whether to show detailed information
            
        Returns:
            DataFrame with historical weather data
            
        Note: This function provides multiple options for historical weather data
        """
        print(f"🌤️ Getting historical weather data from {start_date} to {end_date}")
        print(f"📍 Location: {location}")
        
        # Option 1: Try Open-Meteo (free historical weather API)
        historical_data = self._get_openmeteo_historical(start_date, end_date, detailed)
        
        if len(historical_data) > 0:
            return historical_data
        
        # Fallback: Generate realistic historical weather data
        print("⚠️ Using generated historical weather data (for demo purposes)")
        return self._generate_historical_weather(start_date, end_date)

    def _get_openmeteo_historical(self, start_date: str, end_date: str, detailed: bool) -> pd.DataFrame:
        """
        Get historical weather from Open-Meteo API (free, no key required)
        """
        try:
            # Open-Meteo historical weather API (San Francisco coordinates)
            base_url = "https://archive-api.open-meteo.com/v1/archive"
            
            params = {
                'latitude': 37.7749,      # San Francisco latitude
                'longitude': -122.4194,   # San Francisco longitude
                'start_date': start_date,
                'end_date': end_date,
                'daily': [
                    'temperature_2m_max',
                    'temperature_2m_min', 
                    'temperature_2m_mean',
                    'windspeed_10m_max',
                    'precipitation_sum',
                    'weathercode'
                ],
                'temperature_unit': 'fahrenheit',
                'windspeed_unit': 'mph',
                'precipitation_unit': 'inch',
                'timezone': 'America/Los_Angeles'
            }
            
            print(f"🌐 Fetching from Open-Meteo API...")
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process the response
            daily_data = data['daily']
            dates = daily_data['time']
            
            weather_records = []
            for i, date in enumerate(dates):
                # Convert weather code to description
                weather_code = daily_data['weathercode'][i]
                weather_desc = self._weather_code_to_description(weather_code)
                
                # Calculate average temperature
                temp_max = daily_data['temperature_2m_max'][i]
                temp_min = daily_data['temperature_2m_min'][i]
                temp_avg = (temp_max + temp_min) / 2 if temp_max and temp_min else daily_data['temperature_2m_mean'][i]
                
                weather_records.append({
                    'date': date,
                    'temperature_f': round(temp_avg, 1) if temp_avg else None,
                    'temperature_max_f': temp_max,
                    'temperature_min_f': temp_min,
                    'wind_speed': daily_data['windspeed_10m_max'][i],
                    'precipitation_inch': daily_data['precipitation_sum'][i],
                    'weather_code': weather_code,
                    'forecast': weather_desc,
                    'data_source': 'Open-Meteo'
                })
            
            df = pd.DataFrame(weather_records)
            df['date'] = pd.to_datetime(df['date'])
            
            if detailed:
                print(f"✅ Retrieved {len(df)} days of historical weather data")
                print(f"📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
                print(f"🌡️ Temperature range: {df['temperature_f'].min():.1f}°F to {df['temperature_f'].max():.1f}°F")
            
            return df
            
        except Exception as e:
            print(f"❌ Error fetching Open-Meteo data: {str(e)}")
            return pd.DataFrame()

    def _weather_code_to_description(self, code: int) -> str:
        """Convert WMO weather code to description"""
        weather_codes = {
            0: 'Clear sky',
            1: 'Mainly clear', 2: 'Partly cloudy', 3: 'Overcast',
            45: 'Fog', 48: 'Depositing rime fog',
            51: 'Light drizzle', 53: 'Moderate drizzle', 55: 'Dense drizzle',
            61: 'Slight rain', 63: 'Moderate rain', 65: 'Heavy rain',
            71: 'Slight snow', 73: 'Moderate snow', 75: 'Heavy snow',
            80: 'Slight rain showers', 81: 'Moderate rain showers', 82: 'Violent rain showers',
            95: 'Thunderstorm', 96: 'Thunderstorm with hail'
        }
        return weather_codes.get(code, f'Weather code {code}')

    def _generate_historical_weather(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Generate realistic historical weather data for ML purposes
        Based on San Francisco climate patterns
        """
        print(f"🔧 Generating realistic historical weather data...")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        
        np.random.seed(42)  # For reproducible results
        
        weather_data = []
        for date in dates:
            # San Francisco seasonal temperature pattern
            day_of_year = date.dayofyear
            base_temp = 60 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)  # 50-70°F range
            
            # Add daily variation and some randomness
            temp_variation = np.random.normal(0, 5)
            temperature = base_temp + temp_variation
            
            # Wind speed (SF is windy)
            wind_speed = np.random.normal(12, 4)  # Average ~12 mph
            wind_speed = max(0, wind_speed)
            
            # Precipitation (SF has wet winters, dry summers)
            wet_season_factor = max(0, np.cos(2 * np.pi * (day_of_year - 180) / 365))
            precipitation = np.random.exponential(0.1) * wet_season_factor
            
            # Weather conditions based on precipitation and season
            if precipitation > 0.5:
                weather = 'Rainy'
            elif precipitation > 0.1:
                weather = 'Cloudy'
            elif day_of_year > 120 and day_of_year < 270:  # Summer
                weather = 'Sunny' if np.random.random() > 0.3 else 'Partly Cloudy'
            else:
                weather = 'Partly Cloudy' if np.random.random() > 0.4 else 'Cloudy'
            
            weather_data.append({
                'date': date,
                'temperature_f': round(temperature, 1),
                'temperature_max_f': round(temperature + np.random.uniform(5, 15), 1),
                'temperature_min_f': round(temperature - np.random.uniform(5, 15), 1),
                'wind_speed': round(wind_speed, 1),
                'precipitation_inch': round(precipitation, 2),
                'weather_code': 0,  # Clear sky default
                'forecast': weather,
                'data_source': 'Generated'
            })
        
        df = pd.DataFrame(weather_data)
        print(f"✅ Generated {len(df)} days of weather data")
        return df

    def explore_fred_api(self, api_key: Optional[str] = None, 
                        indicators: Optional[List[str]] = None,
                        detailed: bool = True) -> pd.DataFrame:
        """
        Explore FRED economic indicators API
        
        Args:
            api_key: FRED API key
            indicators: List of FRED series IDs to explore
            detailed: Whether to show detailed information
            
        Returns:
            DataFrame with economic indicators
        """
        api_key = api_key or self.api_keys.get('fred_api_key')
        if not api_key:
            print("⚠️  No FRED API key provided. Get one at: https://research.stlouisfed.org/useraccount/apikey")
            return self._create_sample_fred_data()
        
        indicators = indicators or ['INDPRO', 'CAUR', 'DCOILWTICO', 'GASREGW']
        
        print(f"🔍 Exploring FRED Economic Data API ({len(indicators)} indicators)...")
        
        base_url = "https://api.stlouisfed.org/fred/series/observations"
        all_data = []
        
        indicator_names = {
            'INDPRO': 'Industrial Production Index',
            'CAUR': 'California Unemployment Rate',
            'DCOILWTICO': 'Crude Oil Price (WTI)',
            'GASREGW': 'Regular Gasoline Price'
        }
        
        for series_id in indicators:
            print(f"  📊 Fetching {indicator_names.get(series_id, series_id)}...")
            
            params = {
                'series_id': series_id,
                'api_key': api_key,
                'file_type': 'json',
                'limit': 100000,  # Get ALL available data (API max is usually 100k)
                'sort_order': 'desc'
            }
            
            try:
                response = self.session.get(base_url, params=params)
                response.raise_for_status()
                data = response.json()
                
                for obs in data['observations']:
                    if obs['value'] != '.':
                        all_data.append({
                            'date': obs['date'],
                            'series_id': series_id,
                            'indicator_name': indicator_names.get(series_id, series_id),
                            'value': float(obs['value']),
                            'units': self._get_fred_units(series_id)
                        })
                        
            except Exception as e:
                print(f"    ❌ Error fetching {series_id}: {str(e)}")
                continue
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        
        if detailed and not df.empty:
            self._show_fred_details(df)
            
        return df

    def compare_data_sources(self, eia_df: pd.DataFrame, weather_df: pd.DataFrame, 
                           fred_df: pd.DataFrame) -> None:
        """
        Compare data sources and show relationships
        
        Args:
            eia_df: EIA electricity data
            weather_df: Weather data
            fred_df: Economic indicators data
        """
        print("📊 Comparing Data Sources...")
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('GridSource Data Sources Comparison', fontsize=16)
        
        # Plot 1: EIA generation by fuel type
        if not eia_df.empty:
            eia_pivot = eia_df.pivot_table(
                index='date', columns='fuel_type', 
                values='generation_mwh', aggfunc='sum'
            )
            eia_pivot.plot(kind='area', stacked=True, ax=axes[0,0], alpha=0.7)
            axes[0,0].set_title('Electricity Generation by Fuel Type')
            axes[0,0].set_ylabel('Generation (MWh)')
        
        # Plot 2: Weather temperature
        if not weather_df.empty:
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            weather_daily = weather_df.groupby('date')['temperature_f'].mean()
            weather_daily.plot(ax=axes[0,1], marker='o')
            axes[0,1].set_title('Daily Temperature Forecast')
            axes[0,1].set_ylabel('Temperature (°F)')
        
        # Plot 3: Economic indicators
        if not fred_df.empty:
            for indicator in fred_df['indicator_name'].unique()[:3]:  # Top 3
                indicator_data = fred_df[fred_df['indicator_name'] == indicator]
                axes[1,0].plot(indicator_data['date'], indicator_data['value'], 
                              label=indicator, marker='o')
            axes[1,0].set_title('Economic Indicators')
            axes[1,0].legend()
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # Plot 4: Data availability heatmap
        availability_data = []
        
        if not eia_df.empty:
            eia_dates = set(eia_df['date'].dt.date)
        else:
            eia_dates = set()
            
        if not weather_df.empty:
            weather_dates = set(pd.to_datetime(weather_df['date']).dt.date)
        else:
            weather_dates = set()
            
        if not fred_df.empty:
            fred_dates = set(fred_df['date'].dt.date)
        else:
            fred_dates = set()
        
        all_dates = sorted(eia_dates | weather_dates | fred_dates)
        
        for date in all_dates[-14:]:  # Last 14 days
            availability_data.append({
                'date': date,
                'EIA': 1 if date in eia_dates else 0,
                'Weather': 1 if date in weather_dates else 0,
                'Economic': 1 if date in fred_dates else 0
            })
        
        if availability_data:
            avail_df = pd.DataFrame(availability_data)
            avail_matrix = avail_df.set_index('date')[['EIA', 'Weather', 'Economic']].T
            
            sns.heatmap(avail_matrix, annot=True, cmap='RdYlGn', 
                       cbar_kws={'label': 'Data Available'}, ax=axes[1,1])
            axes[1,1].set_title('Data Availability Heatmap (Last 14 Days)')
        
        plt.tight_layout()
        plt.show()

    def create_sample_dataset(self, days: int = 30) -> Dict[str, pd.DataFrame]:
        """
        Create sample dataset for exploration when APIs are not available
        
        Args:
            days: Number of days of sample data
            
        Returns:
            Dictionary with sample DataFrames
        """
        print(f"🔧 Creating sample dataset for {days} days...")
        
        return {
            'eia': self._create_sample_eia_data(days),
            'weather': self._create_sample_weather_data(days),
            'economic': self._create_sample_fred_data(days)
        }

    def save_exploration_results(self, data: Dict[str, pd.DataFrame], 
                               output_dir: str = "research_output") -> None:
        """
        Save exploration results to CSV files
        
        Args:
            data: Dictionary of DataFrames to save
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"💾 Saving exploration results to {output_dir}/")
        
        for name, df in data.items():
            if not df.empty:
                filename = output_path / f"{name}_sample_data.csv"
                df.to_csv(filename, index=False)
                print(f"  ✅ Saved {name} data: {len(df)} records → {filename}")

    # Helper methods for creating sample data and displaying details
    
    def _create_sample_eia_data(self, days: int) -> pd.DataFrame:
        """Create sample EIA data for testing"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             end=datetime.now(), freq='D')
        
        fuel_types = ['NG', 'SUN', 'WAT', 'WND']
        base_generation = {'NG': 30000, 'SUN': 20000, 'WAT': 15000, 'WND': 10000}
        
        data = []
        for date in dates:
            for fuel in fuel_types:
                # Add realistic variation
                variation = np.random.normal(0, 0.1)
                seasonal = 0.1 * np.sin(2 * np.pi * date.dayofyear / 365)
                
                generation = base_generation[fuel] * (1 + variation + seasonal)
                
                data.append({
                    'date': date,
                    'fuel_type': fuel,
                    'generation_mwh': max(0, generation),
                    'respondent': 'CAL'
                })
        
        return pd.DataFrame(data)

    def _create_sample_weather_data(self, days: int = 7) -> pd.DataFrame:
        """Create sample weather data"""
        dates = pd.date_range(start=datetime.now(), 
                             end=datetime.now() + timedelta(days=days), freq='D')
        
        data = []
        for i, date in enumerate(dates):
            base_temp = 70 + 10 * np.sin(2 * np.pi * date.dayofyear / 365)
            temp_variation = np.random.normal(0, 5)
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'period_name': f'Day {i+1}',
                'temperature_f': int(base_temp + temp_variation),
                'wind_speed': f"{np.random.randint(5, 20)} mph",
                'wind_direction': np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']),
                'forecast': np.random.choice(['Sunny', 'Partly Cloudy', 'Cloudy', 'Rain']),
                'detailed_forecast': 'Sample forecast data for testing'
            })
        
        return pd.DataFrame(data)

    def _create_sample_fred_data(self, days: int = 30) -> pd.DataFrame:
        """Create sample FRED economic data"""
        end_date = datetime.now()
        dates = pd.date_range(start=end_date - timedelta(days=days*7), 
                             end=end_date, freq='W')  # Weekly data
        
        indicators = {
            'INDPRO': ('Industrial Production Index', 102.0, 0.5),
            'CAUR': ('California Unemployment Rate', 4.5, 0.2),
            'DCOILWTICO': ('Crude Oil Price (WTI)', 75.0, 5.0),
            'GASREGW': ('Regular Gasoline Price', 3.50, 0.10)
        }
        
        data = []
        for series_id, (name, base_value, volatility) in indicators.items():
            for date in dates:
                trend = np.random.normal(0, volatility)
                value = base_value + trend
                
                data.append({
                    'date': date,
                    'series_id': series_id,
                    'indicator_name': name,
                    'value': value,
                    'units': self._get_fred_units(series_id)
                })
        
        return pd.DataFrame(data)

    def _show_eia_details(self, df: pd.DataFrame, raw_data: Dict) -> None:
        """Show detailed information about EIA data"""
        print("\n📋 EIA Data Details:")
        print(f"  📅 Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  🏭 Fuel types: {', '.join(df['fuel_type'].unique())}")
        print(f"  📊 Total records: {len(df)}")
        print(f"  ⚡ Total generation: {df['generation_mwh'].sum():,.0f} MWh")
        
        # Show generation by fuel type
        fuel_summary = df.groupby('fuel_type')['generation_mwh'].agg(['count', 'sum', 'mean'])
        print(f"\n  Generation by Fuel Type:")
        for fuel in fuel_summary.index:
            count, total, avg = fuel_summary.loc[fuel]
            print(f"    {fuel}: {total:,.0f} MWh total ({avg:,.0f} avg, {count} records)")

    def _show_weather_details(self, df: pd.DataFrame, raw_data: Dict) -> None:
        """Show detailed information about weather data"""
        print("\n🌤️  Weather Data Details:")
        print(f"  📅 Periods covered: {len(df)}")
        print(f"  🌡️  Temperature range: {df['temperature_f'].min()}°F to {df['temperature_f'].max()}°F")
        print(f"  📊 Forecast types: {', '.join(df['forecast'].unique())}")
        
        # Show periods
        print(f"\n  Forecast Periods:")
        for _, row in df.head().iterrows():
            print(f"    {row['period_name']}: {row['temperature_f']}°F, {row['forecast']}")

    def _show_fred_details(self, df: pd.DataFrame) -> None:
        """Show detailed information about FRED data"""
        print("\n💰 FRED Economic Data Details:")
        print(f"  📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  📊 Total records: {len(df)}")
        print(f"  📈 Indicators: {len(df['indicator_name'].unique())}")
        
        # Show latest values for each indicator
        print(f"\n  Latest Values:")
        latest = df.loc[df.groupby('series_id')['date'].idxmax()]
        for _, row in latest.iterrows():
            print(f"    {row['indicator_name']}: {row['value']:.2f} {row['units']} ({row['date'].date()})")

    def _get_fred_units(self, series_id: str) -> str:
        """Get units for FRED series"""
        units_map = {
            'INDPRO': 'Index 2017=100',
            'CAUR': 'Percent',
            'DCOILWTICO': 'Dollars per Barrel',
            'GASREGW': 'Dollars per Gallon'
        }
        return units_map.get(series_id, 'Units')


def get_synchronized_ml_data(api_keys: Optional[Dict[str, str]] = None, 
                            days: int = 365) -> Dict[str, pd.DataFrame]:
    """
    Get synchronized data from all sources for ML model training
    
    Args:
        api_keys: Dictionary with API keys
        days: Number of historical days to fetch
        
    Returns:
        Dictionary with synchronized DataFrames (same date range)
    """
    explorer = APIExplorer(api_keys)
    
    print(f"🤖 Getting synchronized ML data for {days} days...")
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"📅 Target date range: {start_date.date()} to {end_date.date()}")
    
    # Get EIA data (historical generation)
    print(f"\n1. 📊 EIA Electricity Generation Data:")
    eia_data = explorer.explore_eia_api(days=days, detailed=False)
    
    # Get historical weather data (matching the EIA date range)
    print(f"\n2. 🌤️ Historical Weather Data:")
    weather_data = explorer.get_historical_weather_data(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        detailed=False
    )
    
    # Get FRED economic data (historical)
    print(f"\n3. 💰 FRED Economic Data:")
    fred_data = explorer.explore_fred_api(detailed=False)
    
    # Synchronize date ranges
    print(f"\n🔄 Synchronizing data periods...")
    
    # Find common date range
    eia_dates = set(eia_data['date'].dt.date) if len(eia_data) > 0 else set()
    weather_dates = set(weather_data['date'].dt.date) if len(weather_data) > 0 else set()
    fred_dates = set(fred_data['date'].dt.date) if len(fred_data) > 0 else set()
    
    # Find intersection of all dates
    common_dates = eia_dates & weather_dates & fred_dates
    
    if len(common_dates) == 0:
        print("⚠️ No overlapping dates found. Using largest available range...")
        # Use the range that has the most data
        if len(eia_dates) > 0:
            target_dates = eia_dates
            print(f"   Using EIA date range: {len(target_dates)} days")
        elif len(weather_dates) > 0:
            target_dates = weather_dates
            print(f"   Using weather date range: {len(target_dates)} days")
        else:
            target_dates = fred_dates
            print(f"   Using FRED date range: {len(target_dates)} days")
    else:
        target_dates = common_dates
        print(f"✅ Found {len(common_dates)} overlapping days")
    
    # Filter all data to common dates
    if len(eia_data) > 0:
        eia_data = eia_data[eia_data['date'].dt.date.isin(target_dates)]
    if len(weather_data) > 0:
        weather_data = weather_data[weather_data['date'].dt.date.isin(target_dates)]
    if len(fred_data) > 0:
        fred_data = fred_data[fred_data['date'].dt.date.isin(target_dates)]
    
    # Summary
    print(f"\n📊 Synchronized Data Summary:")
    print(f"  EIA: {len(eia_data):,} records")
    print(f"  Weather: {len(weather_data):,} records")
    print(f"  FRED: {len(fred_data):,} records")
    print(f"  Date range: {len(target_dates):,} days")
    
    return {
        'eia': eia_data,
        'weather': weather_data,
        'economic': fred_data,
        'common_dates': sorted(target_dates)
    }


def quick_explore_apis(api_keys: Optional[Dict[str, str]] = None) -> Dict[str, pd.DataFrame]:
    """
    Quick exploration of all APIs with sample data
    
    Args:
        api_keys: Dictionary with API keys
        
    Returns:
        Dictionary with exploration results
    """
    explorer = APIExplorer(api_keys)
    
    print("🚀 Starting quick API exploration...\n")
    
    # Explore each API
    eia_data = explorer.explore_eia_api(days=7)
    weather_data = explorer.explore_weather_api()
    fred_data = explorer.explore_fred_api()
    
    # Compare data sources
    explorer.compare_data_sources(eia_data, weather_data, fred_data)
    
    return {
        'eia': eia_data,
        'weather': weather_data,
        'economic': fred_data
    }


def analyze_energy_patterns(eia_data: pd.DataFrame, weather_data: pd.DataFrame) -> None:
    """
    Analyze patterns between energy generation and weather
    
    Args:
        eia_data: EIA electricity generation data
        weather_data: Weather forecast data
    """
    print("🔬 Analyzing Energy & Weather Patterns...\n")
    
    if eia_data.empty or weather_data.empty:
        print("❌ Insufficient data for pattern analysis")
        return
    
    # Prepare data
    eia_daily = eia_data.groupby('date')['generation_mwh'].sum().reset_index()
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    weather_daily = weather_data.groupby('date')['temperature_f'].mean().reset_index()
    
    # Merge data
    merged = pd.merge(eia_daily, weather_daily, on='date', how='inner')
    
    if len(merged) < 3:
        print("❌ Not enough overlapping data for analysis")
        return
    
    # Calculate correlation
    correlation = merged['generation_mwh'].corr(merged['temperature_f'])
    
    print(f"📊 Analysis Results:")
    print(f"  🔗 Temperature-Generation Correlation: {correlation:.3f}")
    
    # Create analysis plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Scatter plot
    axes[0].scatter(merged['temperature_f'], merged['generation_mwh'], alpha=0.7)
    axes[0].set_xlabel('Temperature (°F)')
    axes[0].set_ylabel('Total Generation (MWh)')
    axes[0].set_title(f'Temperature vs Generation\n(r = {correlation:.3f})')
    
    # Time series
    axes[1].plot(merged['date'], merged['generation_mwh'], label='Generation', marker='o')
    ax2 = axes[1].twinx()
    ax2.plot(merged['date'], merged['temperature_f'], label='Temperature', 
             color='red', marker='s')
    axes[1].set_ylabel('Generation (MWh)')
    ax2.set_ylabel('Temperature (°F)')
    axes[1].set_title('Generation & Temperature Over Time')
    axes[1].legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("GridSource Data Exploration Helper")
    print("=" * 40)
    
    # Quick exploration with sample data
    results = quick_explore_apis()
    
    # Analyze patterns
    if results['eia'] is not None and results['weather'] is not None:
        analyze_energy_patterns(results['eia'], results['weather'])