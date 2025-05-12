import pandas as pd
import numpy as np
from datetime import datetime


def load_and_clean_data(file_path):
    """
    Загрузка и очистка данных из CSV файла
    """
    # Загрузка данных
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path, parse_dates=['Дата счёта'])
    
    # Переименование колонок для удобства
    column_mapping = {
        'Клиент': 'client_id',
        'Область': 'region',
        'SKU': 'sku',
        'Дата счёта': 'invoice_date',
        'Количество (шт.)': 'quantity',
        'Цена (р.)': 'price',
        'Сумма_заказа': 'final_price',
        'Группа': 'group',
        'Тип': 'type',
        'Категория': 'category'
    }
    
    # Применяем переименование для существующих колонок
    df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns}, inplace=True)
    
    # Обработка числовых колонок
    numeric_columns = ['quantity', 'price', 'final_price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Заполнение пропущенных значений
    df.fillna({
        'quantity': 0,
        'price': 0,
        'final_price': 0
    }, inplace=True)
    
    # Добавление временных компонентов для анализа
    if 'invoice_date' in df.columns:
        df['year'] = df['invoice_date'].dt.year
        df['month'] = df['invoice_date'].dt.month
        df['quarter'] = df['invoice_date'].dt.quarter
        df['date_key'] = df['invoice_date'].dt.to_period('M').dt.start_time
    
    print(f"Loaded {len(df)} records from {file_path}")
    return df


def filter_and_deduplicate_canceled_orders(data, time_window='D'):
    """
    Фильтрует отмененные заказы и удаляет дубликаты в рамках заданного временного окна
    """
    # Копируем данные, чтобы избежать изменения исходного датафрейма
    df = data.copy()
    
    # Фильтруем отмененные заказы, если есть соответствующая колонка
    if 'order_status' in df.columns:
        df = df[df['order_status'] == 'canceled']
    
    # Убедимся, что дата в правильном формате
    date_column = 'Дата счёта' if 'Дата счёта' in df.columns else 'invoice_date'
    
    if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df[date_column] = pd.to_datetime(df[date_column])
    
    # Создаем временное окно для группировки
    df['time_window'] = df[date_column].dt.floor(time_window)
    
    # Определяем ключевые колонки для поиска дубликатов
    client_col = 'Клиент' if 'Клиент' in df.columns else 'client_id'
    sku_col = 'SKU' if 'SKU' in df.columns else 'sku'
    quantity_col = 'Количество' if 'Количество' in df.columns else 'quantity'
    price_col = 'Цена (р.)' if 'Цена (р.)' in df.columns else 'price'
    
    # Удаляем дубликаты, оставляя только одно значение для каждой комбинации
    deduplicated = df.drop_duplicates(
        subset=[client_col, sku_col, quantity_col, price_col, 'time_window']
    )
    
    # Удаляем временную колонку
    deduplicated = deduplicated.drop(columns=['time_window'])
    
    return deduplicated


def consolidate_orders(completed_orders_path, uncompleted_orders_path):
    """
    Консолидация выполненных и невыполненных заказов
    """
    # Загрузка и очистка данных
    completed = load_and_clean_data(completed_orders_path)
    uncompleted = load_and_clean_data(uncompleted_orders_path)
    uncompleted = filter_and_deduplicate_canceled_orders(uncompleted)
    
    # Добавление флага статуса заказа
    completed['order_status'] = 'completed'
    uncompleted['order_status'] = 'uncompleted'
    
    # Консолидация данных
    print("Consolidating orders...")
    consolidated = pd.concat([completed, uncompleted], sort=False)
    consolidated = consolidated.fillna(0)
    consolidated = consolidated.replace('UNK', 'Unk')
    
    # Set final_price to 0 for uncompleted orders
    consolidated.loc[consolidated['order_status'] == 'uncompleted', 'final_price'] = 0
    
    print(f"Consolidated dataset has {len(consolidated)} records")
    return consolidated


def prepare_time_series(data, freq='M', group_by_sku=False):
    """
    Подготовка временных рядов для прогнозирования.
    Может агрегировать данные глобально или по SKU.
    """
    print(f"Preparing time series (group_by_sku={group_by_sku})...")
    data_copy = data.copy()

    if 'order_status' in data_copy.columns:
        data_copy = data_copy[data_copy['order_status'] == 'completed']
        print(f"Filtered to {len(data_copy)} completed orders")

    # Ensure date column is datetime
    if 'invoice_date' not in data_copy.columns:
        raise ValueError("Column 'invoice_date' is missing")
        
    data_copy['invoice_date'] = pd.to_datetime(data_copy['invoice_date'])

    # Create 'date_key' consistently
    if freq == 'M':
        data_copy['date_key'] = data_copy['invoice_date'].dt.to_period('M').dt.start_time
    elif freq == 'W':
        data_copy['date_key'] = data_copy['invoice_date'].dt.to_period('W').dt.start_time
    else:
        data_copy['date_key'] = data_copy['invoice_date'].dt.floor('D')  # Default to daily

    # Add week of month for more features
    data_copy['week_of_month'] = ((data_copy['invoice_date'].dt.day - 1) // 7) + 1

    # Define grouping keys
    grouping_keys = ['date_key']
    if group_by_sku:
        if 'sku' not in data_copy.columns:
            raise ValueError("Column 'sku' is missing for SKU grouping")
        grouping_keys.append('sku')

    # Define aggregations
    agg_dict = {
        'quantity': 'sum',
        'final_price': 'sum',
        'price': 'mean'
    }

    # Add any numeric columns as features with mean aggregation
    for col in data_copy.columns:
        if (col not in agg_dict and col not in grouping_keys and
            col not in ['invoice_date', 'Unnamed: 0', 'client_id'] and
            pd.api.types.is_numeric_dtype(data_copy[col])):
            agg_dict[col] = 'mean'

    # Weekly revenue calculations
    if not group_by_sku:  # Only for global aggregation
        weekly_sums = data_copy.groupby(['date_key', 'week_of_month'])['final_price'].sum().reset_index()
        weekly_pivot = weekly_sums.pivot(
            index='date_key',
            columns='week_of_month',
            values='final_price'
        ).fillna(0)
        
        weekly_pivot = weekly_pivot.rename(columns={
            1: 'week_1_revenue',
            2: 'week_2_revenue',
            3: 'week_3_revenue',
            4: 'week_4_revenue',
            5: 'week_5_revenue'
        })

    # Group and aggregate
    time_series = data_copy.groupby(grouping_keys).agg(agg_dict).reset_index()

    # Rename sku to item_id if grouped by SKU (for AutoGluon compatibility)
    if group_by_sku:
        time_series.rename(columns={'sku': 'item_id'}, inplace=True)
        time_series['item_id'] = time_series['item_id'].astype(str)

    # For global aggregation, merge with weekly data
    if not group_by_sku and 'weekly_pivot' in locals():
        time_series = time_series.merge(weekly_pivot, on='date_key', how='left')
        time_series = time_series.fillna(0)
        
        # Set index only if not grouped by sku
        time_series.set_index('date_key', inplace=True)
        time_series = time_series.sort_index()
    else:
        # Sort by time and item_id (исправляем здесь)
        if group_by_sku:
            # Используем item_id вместо sku для сортировки, так как мы переименовали поле
            time_series = time_series.sort_values(by=['date_key', 'item_id'])
        else:
            time_series = time_series.sort_values(by=['date_key'])

    print(f"Time series preparation complete with shape {time_series.shape}")
    return time_series


def calculate_potential_sales(consolidated_data):
    """
    Расчет потенциальных продаж на основе фактических и упущенных заказов
    """
    print("Calculating potential sales...")
    # Определяем измерения для группировки
    dimensions = ['sku', 'year', 'month', 'quarter', 'region', 'group', 'category']
    existing_dims = [d for d in dimensions if d in consolidated_data.columns]
    
    # Расчет фактических продаж
    completed_sales = consolidated_data[consolidated_data['order_status'] == 'completed'].groupby(
        existing_dims
    ).agg({
        'quantity': 'sum',
        'final_price': 'sum',
        'price': 'mean'
    }).reset_index()
    completed_sales.rename(columns={'final_price': 'actual_revenue'}, inplace=True)
    
    # Расчет упущенных продаж
    lost_sales = consolidated_data[consolidated_data['order_status'] == 'uncompleted'].groupby(
        existing_dims
    ).agg({
        'quantity': 'sum'
    }).reset_index()
    lost_sales.rename(columns={'quantity': 'lost_quantity'}, inplace=True)
    
    # Объединение данных
    potential_data = pd.merge(
        completed_sales, lost_sales, 
        on=existing_dims, 
        how='outer'
    ).fillna(0)
    
    # Расчет потенциальной выручки
    potential_data['lost_revenue'] = potential_data['lost_quantity'] * potential_data['price']
    potential_data['potential_revenue'] = potential_data['actual_revenue'] + potential_data['lost_revenue']
    potential_data['loss_ratio'] = potential_data['lost_revenue'] / potential_data['potential_revenue'].replace(0, np.nan)
    
    print("Potential sales calculation complete")
    return potential_data


def generate_trend_report(data, dimensions, metrics, time_unit='month'):
    """
    Генерация аналитических отчетов по трендам по разным измерениям
    """
    print("Generating trend reports...")
    # Подготовка данных с временным ключом
    if 'date_key' not in data.columns and 'invoice_date' in data.columns:
        if time_unit == 'month':
            data['date_key'] = data['invoice_date'].dt.to_period('M').dt.start_time
        elif time_unit == 'quarter':
            data['date_key'] = data['invoice_date'].dt.to_period('Q').dt.start_time
        elif time_unit == 'year':
            data['date_key'] = data['invoice_date'].dt.to_period('Y').dt.start_time
    
    # Убедимся, что dimensions и metrics - списки
    if isinstance(dimensions, str):
        dimensions = [dimensions]
    if isinstance(metrics, str):
        metrics = [metrics]
    
    # Словарь для хранения отчетов
    reports = {}
    
    # Генерация отчетов для каждого измерения
    for dimension in dimensions:
        # Проверка наличия измерения в данных
        if dimension not in data.columns:
            continue
        
        # Получение уникальных значений измерения
        values = data[dimension].unique()
        
        # Словарь для хранения трендов по значениям измерения
        dimension_trends = {}
        
        # Анализ тренда для каждого значения измерения
        for value in values:
            value_data = data[data[dimension] == value]
            
            # Группировка по времени и расчет метрик
            trend = value_data.groupby('date_key').agg({
                metric: 'sum' for metric in metrics if metric in data.columns
            }).reset_index()
            
            # Сортировка по времени
            trend = trend.sort_values('date_key')
            
            # Сохранение тренда
            dimension_trends[value] = trend
        
        # Сохранение отчета по измерению
        reports[dimension] = dimension_trends
    
    print(f"Generated trend reports for {len(reports)} dimensions")
    return reports


def get_top_performers(data, dimension, metric, n=10, ascending=False):
    """
    Получение топ-N значений по заданному измерению и метрике
    """
    # Проверка наличия колонок
    if dimension not in data.columns or metric not in data.columns:
        raise ValueError(f"Columns '{dimension}' or '{metric}' not found in data")
    
    # Группировка и расчет суммы метрики
    summary = data.groupby(dimension).agg({
        metric: 'sum'
    }).reset_index()
    
    # Сортировка и выбор топ-N
    top_n = summary.sort_values(metric, ascending=ascending).head(n)
    
    return top_n 