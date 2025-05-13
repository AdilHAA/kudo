import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к директории src для импорта модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_processing import (
    load_and_clean_data,
    consolidate_orders,
    prepare_time_series,
    calculate_potential_sales,
    generate_trend_report,
    get_top_performers,
    save_data
)

from src.forecasting import (
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
        completed_data_path = os.path.join('data', 'final_df_latest2.parquet')
        uncompleted_data_path = os.path.join('data', 'denied.parquet')
        
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
        
        # Сохраняем результаты в папку results
        results_dir = 'results'
        os.makedirs(results_dir, exist_ok=True)
        potential_sales.sort_values('potential_revenue', ascending=False).head(10).to_parquet(
            os.path.join(results_dir, "potential_sales.parquet")
        )
        
        # 3. Sales forecasting
        print("\n3. Sales forecasting...")
        
        # Choose forecasting method
        # Methods: 'prophet', 'xgboost', 'autogluon', 'average'
        forecast_method = 'autogluon'  # Используем метод autogluon
        print(f"Using {forecast_method} forecasting method")
        
        # Список целевых переменных для прогнозирования
        target_variables = ['final_price', 'quantity']
        
        # Каталог для моделей
        models_dir = 'models'
        os.makedirs(models_dir, exist_ok=True)
        
        # Forecasting with AutoGluon (uses SKU-level data)
        for target_variable in target_variables:
            print(f"\n--- Forecasting for target: {target_variable} ---")
            # Prepare data grouped by SKU for AutoGluon
            print("Preparing time series data with SKU grouping...")
            ts_data_sku = prepare_time_series(consolidated_data, freq='M', group_by_sku=True)
            
            # If there are too many SKUs, we might want to filter to top ones
            unique_skus = ts_data_sku['item_id'].nunique()
            print(f"Found {unique_skus} unique SKUs")
            
            # Фильтруем до топ-200 SKU
            print(f"Filtering to focus on top 200 SKUs...")
            # Get top SKUs by the current target variable
            top_skus = get_top_performers(consolidated_data, 'sku', target_variable, n=200)
            
            # Обеспечиваем соответствие форматов
            # Выводим первые несколько строк для отладки
            print(f"Top SKUs (first 5): {top_skus['sku'].head().tolist()}")
            print(f"Time series item_id (first 5): {ts_data_sku['item_id'].head().tolist()}")
            
            # Преобразуем идентификаторы к одному формату для правильного сопоставления
            top_sku_list = top_skus['sku'].astype(str).tolist()
            
            # Преобразуем item_id к такому же формату как sku для сравнения
            ts_data_sku['item_id_str'] = ts_data_sku['item_id'].astype(str)
            
            # Фильтруем временные ряды
            filtered_ts = ts_data_sku[ts_data_sku['item_id_str'].isin(top_sku_list)]
            
            # Проверяем результаты фильтрации
            print(f"До фильтрации: {len(ts_data_sku)} строк")
            print(f"После фильтрации: {len(filtered_ts)} строк")
            print(f"Количество уникальных SKU после фильтрации: {filtered_ts['item_id'].nunique()}")
            
            # Если после фильтрации нет данных или очень мало, используем больше SKU
            if len(filtered_ts) < 100 or filtered_ts['item_id'].nunique() < 5:
                print("Слишком мало данных после фильтрации, используем все доступные SKU")
                filtered_ts = ts_data_sku
            
            # Удаляем временный столбец
            if 'item_id_str' in filtered_ts.columns:
                filtered_ts = filtered_ts.drop('item_id_str', axis=1)
            
            # Сохраняем подготовленные данные для отладки
            debug_file = os.path.join(results_dir, f"debug_ts_data_{target_variable}.parquet")
            try:
                # Преобразуем все строковые столбцы в тип 'string' для pyarrow
                ts_to_save = filtered_ts.copy()
                for col in ts_to_save.columns:
                    if ts_to_save[col].dtype == 'object':
                        ts_to_save[col] = ts_to_save[col].astype('string')
                
                save_data(ts_to_save, debug_file)
                print(f"Saved debug data to {debug_file}")
            except Exception as e:
                print(f"Warning: Could not save debug data: {e}")
            
            # Run the AutoGluon forecast
            print(f"Running AutoGluon forecast for {target_variable}...")
            try:
                forecast_result = forecast_sales(
                    filtered_ts,
                    periods=12,
                    method=forecast_method,
                    target=target_variable,
                    item_id_col='item_id',
                    timestamp_col='date_key',
                    models_dir=models_dir  # Передаем путь к каталогу для моделей
                )
                
                # Process and save the results
                if not forecast_result.empty:
                    print("Saving forecast results...")
                    try:
                        # Преобразуем все строковые столбцы в тип 'string' для pyarrow
                        forecast_to_save = forecast_result.copy()
                        for col in forecast_to_save.columns:
                            if forecast_to_save[col].dtype == 'object':
                                forecast_to_save[col] = forecast_to_save[col].astype('string')
                        
                        save_data(forecast_to_save, os.path.join(results_dir, f"forecast_{forecast_method}_{target_variable}.parquet"))
                        print(f"Saved forecast results to: forecast_{forecast_method}_{target_variable}.parquet")
                        
                        # Aggregate across all SKUs if needed
                        if 'item_id' in forecast_result.columns:
                            print("Aggregating forecast across all SKUs...")
                            agg_cols = [col for col in forecast_result.columns 
                                      if col not in ['item_id', 'timestamp']]
                            agg_forecast = forecast_result.groupby('timestamp')[agg_cols].sum().reset_index()
                            
                            # Сохраняем агрегированный прогноз
                            save_data(agg_forecast, os.path.join(results_dir, f"agg_forecast_{forecast_method}_{target_variable}.parquet"))
                            print(f"Saved aggregate forecast to: agg_forecast_{forecast_method}_{target_variable}.parquet")
                            print(f"\nAggregate forecast for {target_variable} across all SKUs:")
                            print(agg_forecast)
                    except Exception as e:
                        print(f"Error saving forecast: {e}")
                else:
                    print(f"Warning: Empty forecast results for {target_variable}")
                    # Попробуем использовать среднее значение
                    print("Falling back to average forecast method...")
                    time_series_global = prepare_time_series(consolidated_data, freq='M', group_by_sku=False)
                    fallback_forecast = forecast_sales(
                        time_series_global,
                        periods=12,
                        method='average',
                        target=target_variable
                    )
                    if not fallback_forecast.empty:
                        fallback_to_save = fallback_forecast.copy()
                        if isinstance(fallback_to_save.index, pd.DatetimeIndex):
                            fallback_to_save = fallback_to_save.reset_index()
                        save_data(fallback_to_save, os.path.join(results_dir, f"forecast_fallback_{target_variable}.parquet"))
                        print(f"Saved fallback forecast to: forecast_fallback_{target_variable}.parquet")
            except Exception as e:
                print(f"Error in forecasting with AutoGluon: {e}")
                import traceback
                traceback.print_exc()
                print("Falling back to average forecast method...")
                time_series_global = prepare_time_series(consolidated_data, freq='M', group_by_sku=False)
                fallback_forecast = forecast_sales(
                    time_series_global,
                    periods=12,
                    method='average',
                    target=target_variable
                )
                if not fallback_forecast.empty:
                    fallback_to_save = fallback_forecast.copy()
                    if isinstance(fallback_to_save.index, pd.DatetimeIndex):
                        fallback_to_save = fallback_to_save.reset_index()
                    save_data(fallback_to_save, os.path.join(results_dir, f"forecast_fallback_{target_variable}.parquet"))
                    print(f"Saved fallback forecast to: forecast_fallback_{target_variable}.parquet")
        
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
        save_data(top_clients, os.path.join(results_dir, "top_clients.parquet"))
        
        # Top SKUs by quantity
        top_skus = get_top_performers(consolidated_data, 'sku', 'quantity')
        print("\nTop 10 SKUs by quantity:")
        print(top_skus)
        save_data(top_skus, os.path.join(results_dir, "top_skus.parquet"))
        
        print("\nAnalysis complete! Results saved to parquet files.")
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 