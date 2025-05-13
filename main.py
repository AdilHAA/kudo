import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from data_processing import (
    load_and_clean_data,
    consolidate_orders,
    prepare_time_series,
    calculate_potential_sales,
    generate_trend_report,
    get_top_performers
)

from forecasting import (
    forecast_sales,
    forecast_by_segment
)


def main():
    """
    Main function to run the sales analysis and forecasting
    """
    try:
        print("=" * 80)
        print("SALES ANALYSIS AND FORECASTING".center(80))
        print("=" * 80)
        
        # 1. Load and consolidate data
        print("\n1. Loading and consolidating data...")
        completed_data_path = 'final_df_latest2.parquet'
        uncompleted_data_path = 'denied.parquet'
        
        # Check if files exist
        if not os.path.exists(completed_data_path):
            raise FileNotFoundError(f"File not found: {completed_data_path}")
        if not os.path.exists(uncompleted_data_path):
            raise FileNotFoundError(f"File not found: {uncompleted_data_path}")
        
        consolidated_data = consolidate_orders(completed_data_path, uncompleted_data_path)
        print(f"Loaded {len(consolidated_data)} records in total")
        
        # 2. Calculate potential sales
        print("\n2. Calculating potential sales...")
        potential_sales = calculate_potential_sales(consolidated_data)
        print("Top 5 SKUs by potential revenue:")
        print(potential_sales.sort_values('potential_revenue', ascending=False).head())
        potential_sales.sort_values('potential_revenue', ascending=False).head().to_parquet("potential_sales.parquet")
        
        # 3. Sales forecasting
        print("\n3. Sales forecasting...")
        
        # Choose forecasting method
        # Methods: 'prophet', 'xgboost', 'autogluon', 'average'
        forecast_method = 'autogluon'
        print(f"Using {forecast_method} forecasting method")
        
        # Список целевых переменных для прогнозирования
        target_variables = ['final_price', 'quantity']
        
        # Forecasting with AutoGluon (uses SKU-level data)
        if forecast_method == 'autogluon':
            # Для каждой целевой переменной делаем прогнозирование
            for target_variable in target_variables:
                print(f"\n--- Forecasting for target: {target_variable} ---")
                # Prepare data grouped by SKU for AutoGluon
                print("Preparing time series data with SKU grouping...")
                ts_data_sku = prepare_time_series(consolidated_data, freq='M', group_by_sku=True)
                
                # If there are too many SKUs, we might want to filter to top ones
                unique_skus = ts_data_sku['item_id'].nunique()
                print(f"Found {unique_skus} unique SKUs")
                
                if unique_skus > 150:  # Увеличено с 50 до 150
                    print(f"Filtering to focus on top 150 SKUs...")  # Увеличено с 50 до 150
                    # Get top SKUs by the current target variable
                    top_skus = get_top_performers(consolidated_data, 'sku', target_variable, n=150)  # Увеличено с 50 до 150
                    top_sku_list = top_skus['sku'].astype(str).tolist()
                    print(f"Filtered to top {len(top_sku_list)} SKUs")
                    ts_data_sku = ts_data_sku[ts_data_sku['item_id'].isin(top_sku_list)]
                
                # Run the AutoGluon forecast
                print(f"Running AutoGluon forecast for {target_variable}...")
                forecast_result = forecast_sales(
                    ts_data_sku,
                    periods=12,
                    method=forecast_method,
                    target=target_variable,
                    item_id_col='item_id',
                    timestamp_col='date_key'
                )
                
                # Process and save the results
                if not forecast_result.empty:
                    print("Saving forecast results...")
                    forecast_result.to_parquet(f"forecast_{forecast_method}_{target_variable}.parquet", index=False)
                    
                    # Aggregate across all SKUs if needed
                    if 'item_id' in forecast_result.columns:
                        print("Aggregating forecast across all SKUs...")
                        agg_cols = [col for col in forecast_result.columns 
                                   if col not in ['item_id', 'timestamp']]
                        agg_forecast = forecast_result.groupby('timestamp')[agg_cols].sum().reset_index()
                        agg_forecast.to_parquet(f"agg_forecast_{forecast_method}_{target_variable}.parquet", index=False)
                        print(f"\nAggregate forecast for {target_variable} across all SKUs:")
                        print(agg_forecast)
                else:
                    print("No forecast results were returned")
        
        # Traditional forecasting methods (global data)
        else:
            # Для каждой целевой переменной делаем прогнозирование
            for target_variable in target_variables:
                print(f"\n--- Forecasting for target: {target_variable} with {forecast_method} ---")
                # Prepare global time series (not grouped by SKU)
                print("Preparing global time series data...")
                time_series_global = prepare_time_series(consolidated_data, freq='M', group_by_sku=False)
                
                # Save the prepared time series
                if isinstance(time_series_global.index, pd.DatetimeIndex):
                    time_series_global.reset_index().to_parquet(f"time_series_global_{target_variable}.parquet", index=False)
                else:
                    time_series_global.to_parquet(f"time_series_global_{target_variable}.parquet", index=False)
                
                # Run the forecast
                print(f"Running {forecast_method} forecast...")
                forecast_result = forecast_sales(
                    time_series_global,
                    periods=12,
                    method=forecast_method,
                    target=target_variable
                )
                
                # Process and save results
                if not forecast_result.empty:
                    print("Saving forecast results...")
                    if isinstance(forecast_result.index, pd.DatetimeIndex):
                        forecast_result.reset_index().to_parquet(f"forecast_{forecast_method}_{target_variable}.parquet", index=False)
                    else:
                        forecast_result.to_parquet(f"forecast_{forecast_method}_{target_variable}.parquet", index=False)
                    
                    print(f"\nForecast for {target_variable} next 12 months:")
                    if hasattr(forecast_result, 'tail'):
                        print(forecast_result.tail(12))
                    else:
                        print(forecast_result)
                else:
                    print(f"No forecast results were returned for {target_variable}")
        
        # 4. Category-level forecasting (optional)
        run_category_forecasts = True
        if run_category_forecasts:
            # Для каждой целевой переменной делаем прогнозирование по категориям
            for target_variable in target_variables:
                print(f"\n4. Forecasting {target_variable} by product category...")
                category_forecasts = forecast_by_segment(
                    consolidated_data,
                    'category',
                    periods=12,
                    method=forecast_method,
                    target=target_variable
                )
                
                # Save category forecasts
                for category, forecast in category_forecasts.items():
                    if not forecast.empty:
                        print(f"\nForecast for category: {category}")
                        print(forecast.tail(12) if hasattr(forecast, 'tail') else forecast)
                        
                        # Save to CSV
                        filename = f"forecast_{category}_{forecast_method}_{target_variable}.parquet"
                        filename = filename.replace(" ", "_").replace("/", "_")
                        if isinstance(forecast.index, pd.DatetimeIndex):
                            forecast.reset_index().to_parquet(filename, index=False)
                        else:
                            forecast.to_parquet(filename, index=False)
        
        # 5. Generate trend reports
        print("\n5. Generating trend reports...")
        trend_reports = generate_trend_report(
            consolidated_data,
            dimensions=['category', 'group', 'region'],
            metrics=['final_price', 'quantity']  # Всегда включаем обе метрики
        )
        print(f"Generated trend reports for {len(trend_reports)} dimensions")
        
        # 6. Identify top performers
        print("\n6. Identifying top performers...")
        
        # Top clients by revenue
        top_clients = get_top_performers(consolidated_data, 'client_id', 'final_price')
        print("\nTop 10 clients by revenue:")
        print(top_clients)
        top_clients.to_parquet("top_clients.parquet")
        
        # Top SKUs by quantity
        top_skus = get_top_performers(consolidated_data, 'sku', 'quantity')
        print("\nTop 10 SKUs by quantity:")
        print(top_skus)
        top_skus.to_parquet("top_skus.parquet")
        
        print("\nAnalysis complete! Results saved to Parquet files.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 