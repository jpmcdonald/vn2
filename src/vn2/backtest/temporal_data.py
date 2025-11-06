"""
Temporal Data Manager for backtesting with strict time constraints.

This module ensures that backtests respect temporal data availability,
preventing any future data leakage that would invalidate research results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class DataSnapshot:
    """Data available at a specific decision epoch."""
    decision_week: int
    decision_date: pd.Timestamp
    available_demand: pd.DataFrame
    available_state: Dict
    available_sales: Dict[int, pd.DataFrame]  # week -> sales data
    forecast_cutoff: pd.Timestamp


class TemporalDataManager:
    """
    Manages data availability based on decision epoch.
    
    Ensures strict temporal constraints:
    - Week N decision can only use data available before Week N start
    - No future sales data allowed
    - Forecasts must be trained only on historical data
    """
    
    def __init__(self, data_dir: Path):
        """
        Initialize with data directory.
        
        Args:
            data_dir: Path to data directory containing raw weekly files
        """
        self.data_dir = data_dir
        self.raw_dir = data_dir / 'raw'
        self.states_dir = data_dir / 'states'
        self.submissions_dir = data_dir / 'submissions'
        
        # Competition timeline
        self.week_dates = {
            0: pd.Timestamp('2024-04-08'),  # Initial state
            1: pd.Timestamp('2024-04-15'),  # Week 1 decision
            2: pd.Timestamp('2024-04-22'),  # Week 2 decision  
            3: pd.Timestamp('2024-04-29'),  # Week 3 decision
            4: pd.Timestamp('2024-05-06'),  # Week 4 decision
            5: pd.Timestamp('2024-05-13'),  # Week 5 decision (estimated)
            6: pd.Timestamp('2024-05-20'),  # Week 6 decision (estimated)
        }
        
        # Load all available data files
        self._load_available_files()
    
    def _load_available_files(self) -> None:
        """Load and catalog all available data files."""
        self.available_files = {
            'initial_state': self.raw_dir / 'Week 0 - 2024-04-08 - Initial State.csv',
            'week0_sales': self.raw_dir / 'Week 0 - 2024-04-08 - Sales.csv',
            'week1_sales': self.raw_dir / 'Week 1 - 2024-04-15 - Sales.csv', 
            'week2_sales': self.raw_dir / 'Week 2 - 2024-04-22 - Sales.csv',
            'week3_sales': self.raw_dir / 'Week 3 - 2024-04-29 - Sales.csv',
            'week4_sales': self.raw_dir / 'Week 4 - 2024-05-06 - Sales.csv',
        }
        
        # State files
        for week in range(1, 5):
            state_file = self.states_dir / f'state{week}.csv'
            if state_file.exists():
                self.available_files[f'state{week}'] = state_file
        
        # Submission files
        for week in range(1, 7):
            order_file = self.submissions_dir / f'order{week}_jpatrickmcdonald.csv'
            if order_file.exists():
                self.available_files[f'order{week}'] = order_file
    
    def get_data_snapshot(self, decision_week: int) -> DataSnapshot:
        """
        Get data snapshot available at decision time.
        
        Args:
            decision_week: Week number (1-6)
        
        Returns:
            DataSnapshot with all data available at decision time
        """
        decision_date = self.week_dates[decision_week]
        
        # Historical demand data (up to but not including decision week)
        available_demand = self._get_historical_demand(decision_week)
        
        # State data (current inventory position)
        available_state = self._get_current_state(decision_week)
        
        # Sales data (actual demand for completed weeks)
        available_sales = self._get_available_sales(decision_week)
        
        # Forecast training cutoff (can't use data after this)
        forecast_cutoff = self._get_forecast_cutoff(decision_week)
        
        return DataSnapshot(
            decision_week=decision_week,
            decision_date=decision_date,
            available_demand=available_demand,
            available_state=available_state,
            available_sales=available_sales,
            forecast_cutoff=forecast_cutoff
        )
    
    def _get_historical_demand(self, decision_week: int) -> pd.DataFrame:
        """
        Get historical demand data available at decision time.
        
        For research validity, we can only use demand data that would have
        been available when making the decision.
        """
        # Load the processed demand data
        demand_path = self.data_dir / 'processed' / 'demand_long.parquet'
        
        if not demand_path.exists():
            raise FileNotFoundError(f"Demand data not found: {demand_path}")
        
        demand_df = pd.read_parquet(demand_path)
        
        # Ensure week column is datetime
        if 'week' in demand_df.columns:
            demand_df['week'] = pd.to_datetime(demand_df['week'])
        
        # Filter to only data available before decision date
        cutoff_date = self.week_dates[decision_week]
        available_demand = demand_df[demand_df['week'] < cutoff_date].copy()
        
        return available_demand
    
    def _get_current_state(self, decision_week: int) -> Dict:
        """
        Get current inventory state at decision time.
        
        Args:
            decision_week: Week number
        
        Returns:
            Dict with inventory state per SKU
        """
        if decision_week == 1:
            # Use initial state
            state_file = self.available_files['initial_state']
        else:
            # Use previous week's ending state
            state_file = self.available_files.get(f'state{decision_week-1}')
        
        if not state_file or not state_file.exists():
            raise FileNotFoundError(f"State file not found for decision week {decision_week}")
        
        state_df = pd.read_csv(state_file)
        
        # Convert to dictionary for easy lookup
        state_dict = {}
        for _, row in state_df.iterrows():
            if decision_week == 1:
                # Initial state format
                key = (int(row['Store']), int(row['Product']))
                state_dict[key] = {
                    'on_hand': int(row.get('In Stock', 0)),
                    'intransit_1': 0,  # No orders in transit initially
                    'intransit_2': 0
                }
            else:
                # State file format
                key = (int(row['Store']), int(row['Product']))
                state_dict[key] = {
                    'on_hand': int(row.get('End Inventory', 0)),
                    'intransit_1': int(row.get('In Transit W+1', 0)),
                    'intransit_2': int(row.get('In Transit W+2', 0))
                }
        
        return state_dict
    
    def _get_available_sales(self, decision_week: int) -> Dict[int, pd.DataFrame]:
        """
        Get actual sales data for completed weeks.
        
        Args:
            decision_week: Current decision week
        
        Returns:
            Dict mapping week number to sales DataFrame
        """
        available_sales = {}
        
        # Can only use sales from completed weeks
        for week in range(1, decision_week):
            sales_file = self.available_files.get(f'week{week}_sales')
            if sales_file and sales_file.exists():
                sales_df = pd.read_csv(sales_file)
                available_sales[week] = sales_df
        
        return available_sales
    
    def _get_forecast_cutoff(self, decision_week: int) -> pd.Timestamp:
        """
        Get latest date that can be used for training forecasts.
        
        Forecasts must be trained only on data available before decision time.
        """
        # Use data up to (but not including) the decision week
        return self.week_dates[decision_week] - timedelta(days=1)
    
    def validate_temporal_constraints(
        self, 
        data_used: List[str], 
        decision_week: int
    ) -> Tuple[bool, List[str]]:
        """
        Validate that no future data was used.
        
        Args:
            data_used: List of data sources used
            decision_week: Week of decision
        
        Returns:
            (is_valid, violations): Validation result and list of violations
        """
        violations = []
        decision_date = self.week_dates[decision_week]
        
        for data_source in data_used:
            # Check if data source contains future information
            if f'week{decision_week}' in data_source.lower() and 'sales' in data_source.lower():
                violations.append(f"Used {data_source} for Week {decision_week} decision (future data)")
            
            # Check for any data from future weeks
            for future_week in range(decision_week + 1, 7):
                if f'week{future_week}' in data_source.lower():
                    violations.append(f"Used {data_source} for Week {decision_week} decision (future week {future_week})")
        
        return len(violations) == 0, violations
    
    def get_submission_orders(self, week: int) -> Optional[pd.DataFrame]:
        """
        Get submitted orders for a specific week.
        
        Args:
            week: Week number
        
        Returns:
            DataFrame with orders or None if not available
        """
        order_file = self.available_files.get(f'order{week}')
        if order_file and order_file.exists():
            return pd.read_csv(order_file)
        return None
    
    def get_week_sales(self, week: int) -> Optional[pd.DataFrame]:
        """
        Get actual sales for a specific week.
        
        Args:
            week: Week number
        
        Returns:
            DataFrame with sales or None if not available
        """
        sales_file = self.available_files.get(f'week{week}_sales')
        if sales_file and sales_file.exists():
            return pd.read_csv(sales_file)
        return None
    
    def create_temporal_forecast_data(
        self, 
        decision_week: int,
        demand_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create forecast training data respecting temporal constraints.
        
        Args:
            decision_week: Week of decision
            demand_df: Full demand DataFrame
        
        Returns:
            DataFrame with only temporally valid data for training
        """
        cutoff_date = self._get_forecast_cutoff(decision_week)
        
        # Filter demand data
        valid_data = demand_df[demand_df['week'] <= cutoff_date].copy()
        
        return valid_data
    
    def get_available_weeks(self) -> List[int]:
        """Get list of weeks for which we have data."""
        weeks = []
        for week in range(1, 7):
            if self.week_dates.get(week) is not None:
                weeks.append(week)
        return weeks
    
    def print_data_availability(self) -> None:
        """Print summary of available data files."""
        print("Data Availability Summary:")
        print("="*50)
        
        for key, path in self.available_files.items():
            status = "✅" if path.exists() else "❌"
            print(f"{status} {key}: {path}")
        
        print()
        print("Week Timeline:")
        for week, date in self.week_dates.items():
            print(f"Week {week}: {date.strftime('%Y-%m-%d')}")
