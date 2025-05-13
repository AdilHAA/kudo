import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    """
    Загрузка и очистка данных из файла (parquet или csv)
    """
    # Загрузка данных
    if file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
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
    
    # Проверяем, является ли столбец invoice_date типом datetime
    if 'invoice_date' in df.columns:
        # Преобразуем в datetime, если не является
        if not pd.api.types.is_datetime64_any_dtype(df['invoice_date']):
            df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
            
        # Добавление временных компонентов для анализа
        df['year'] = df['invoice_date'].dt.year
        df['month'] = df['invoice_date'].dt.month
        df['quarter'] = df['invoice_date'].dt.quarter
        df['date_key'] = df['invoice_date'].dt.to_period('M').dt.start_time
    
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
    consolidated = pd.concat([completed, uncompleted], sort=False)
    consolidated = consolidated.fillna(0)
    consolidated = consolidated.replace('UNK', 'Unk')
    
    # Установка значения final_price на 0 для невыполненных заказов
    consolidated.loc[consolidated['order_status'] == 'uncompleted', 'final_price'] = 0
    
    return consolidated


def prepare_time_series(data, freq='M', group_by_sku=False):
    """
    Подготовка временных рядов для прогнозирования
    
    Параметры:
        data (DataFrame): датафрейм с данными заказов
        freq (str): частота временного ряда ('D' для дней, 'W' для недель, 'M' для месяцев)
        group_by_sku (bool): если True, то данные группируются по SKU для многошагового прогнозирования
        
    Возвращает:
        DataFrame: временной ряд, подготовленный для прогнозирования
    """
    # Копируем данные, чтобы избежать изменений во входных данных
    data_copy = data.copy()
    
    # Если данные нужно сгруппировать по SKU (для AutoGluon)
    if group_by_sku:
        # Выбираем только выполненные заказы
        if 'order_status' in data_copy.columns:
            data_copy = data_copy[data_copy['order_status'] == 'completed']
        
        # Убедимся, что у нас есть колонка с датой
        if 'date_key' not in data_copy.columns and 'invoice_date' in data_copy.columns:
            data_copy['date_key'] = data_copy['invoice_date'].dt.to_period(freq).dt.start_time
        
        # Группировка данных по SKU и дате
        grouped_data = data_copy.groupby(['sku', 'date_key']).agg({
            'quantity': 'sum',
            'final_price': 'sum',
            'price': 'mean'
        }).reset_index()
        
        # Переименовываем колонки для AutoGluon
        grouped_data = grouped_data.rename(columns={'sku': 'item_id'})
        
        return grouped_data
        
    else:
        # Группировка данных по дате
        if 'order_status' in data_copy.columns:
            data_copy = data_copy[data_copy['order_status'] == 'completed']
        
        # Убедимся, что у нас есть колонка с датой
        if 'date_key' not in data_copy.columns and 'invoice_date' in data_copy.columns:
            data_copy['date_key'] = data_copy['invoice_date'].dt.to_period(freq).dt.start_time
        
        # Добавляем расчет недели месяца
        data_copy['week_of_month'] = ((data_copy['invoice_date'].dt.day - 1) // 7) + 1
        
        # Функция для нахождения наиболее частого значения
        def most_frequent(series):
            if series.empty:
                return None
            counts = series.value_counts()
            if counts.empty:
                return None
            return counts.index[0]
        
        # Создадим словарь агрегаций
        agg_dict = {}
        
        # Добавляем базовые агрегации если соответствующие столбцы существуют
        if 'quantity' in data_copy.columns:
            agg_dict['quantity'] = 'sum'
        if 'final_price' in data_copy.columns:
            agg_dict['final_price'] = 'sum'
        
        # Определяем типы столбцов и добавляем соответствующие агрегации
        for col in data_copy.columns:
            # Пропускаем уже обработанные столбцы и служебные
            if col in agg_dict or col in ['date_key', 'order_status', 'invoice_date', 'Unnamed: 0', 
                                         'Дата год-месяц', 'client_id', 'sku', 'time_idx', 
                                         'days_until_holiday', 'is_holiday', 'day_of_week', 'week_of_month']:
                continue
            
            # Определяем тип данных и выбираем подходящую агрегацию
            dtype = data_copy[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):  # Для числовых колонок
                agg_dict[col] = 'max'
            elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):  # Для категориальных
                agg_dict[col] = most_frequent
            elif pd.api.types.is_datetime64_dtype(dtype):  # Для дат
                agg_dict[col] = 'max'
            elif pd.api.types.is_bool_dtype(dtype):  # Для булевых
                agg_dict[col] = 'any'
        
        # Вычисляем суммарную прибыль по неделям месяца
        weekly_sums = data_copy.groupby(['date_key', 'week_of_month'])['final_price'].sum().reset_index()
        
        # Преобразуем в широкий формат
        weekly_pivot = weekly_sums.pivot(
            index='date_key', 
            columns='week_of_month', 
            values='final_price'
        ).fillna(0)
        
        # Переименовываем числовые столбцы в строковые
        weekly_pivot = weekly_pivot.rename(columns={
            1: 'week_1_revenue',
            2: 'week_2_revenue',
            3: 'week_3_revenue',
            4: 'week_4_revenue',
            5: 'week_5_revenue'
        })
        
        # Группировка данных по периоду времени с учетом всех агрегаций
        time_series = data_copy.groupby('date_key').agg(agg_dict).reset_index()
        time_series = time_series.merge(weekly_pivot, on='date_key', how='left')
        time_series = time_series.fillna(0)
        
        # Установка даты в качестве индекса
        time_series.set_index('date_key', inplace=True)
        
        # Убедимся, что индекс отсортирован
        time_series = time_series.sort_index()
        return time_series


def calculate_potential_sales(consolidated_data):
    """
    Расчет потенциальных продаж на основе фактических и упущенных заказов
    """
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
    
    return potential_data


def generate_trend_report(data, dimensions, metrics, time_unit='month'):
    """
    Генерация аналитических отчетов по трендам по разным измерениям
    """
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
    
    return reports


def get_top_performers(data, dimension, metric, n=10, ascending=False):
    """
    Получение топ-N значений по заданному измерению и метрике
    """
    # Проверка наличия колонок
    if dimension not in data.columns or metric not in data.columns:
        raise ValueError(f"Колонки '{dimension}' или '{metric}' не найдены в данных")
    
    # Группировка и расчет суммы метрики
    summary = data.groupby(dimension).agg({
        metric: 'sum'
    }).reset_index()
    
    # Сортировка и выбор топ-N
    top_n = summary.sort_values(metric, ascending=ascending).head(n)
    
    return top_n


def save_data(df, file_path, index=False):
    """
    Сохраняет DataFrame в формат Parquet (или CSV в зависимости от расширения файла)
    
    Параметры:
        df (DataFrame): DataFrame для сохранения
        file_path (str): Путь для сохранения
        index (bool): Сохранять ли индекс
    """
    # Создаем директорию, если она не существует
    import os
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    # Сохраняем в нужном формате
    if file_path.endswith('.parquet'):
        df.to_parquet(file_path, index=index)
    else:
        df.to_csv(file_path, index=index)
    
    return file_path 