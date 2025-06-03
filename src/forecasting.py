import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os
warnings.filterwarnings('ignore')


def get_feature_importance(model, X, importance_type='gain', return_shap_values=False, plot=False, figsize=(12, 8)):
    """
    Получает важность признаков из модели XGBoost, используя как встроенные методы, так и SHAP.
    """
    import xgboost as xgb
    import shap
    
    # Определяем имена признаков
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
    else:
        feature_names = [f'feature_{i}' for i in range(X.shape[1])]
    
    # Получаем важность признаков из XGBoost
    if isinstance(model, xgb.Booster):
        xgb_importance_dict = model.get_score(importance_type=importance_type)
        # Не все признаки могут быть в словаре, если они не использовались
        xgb_importance = np.zeros(len(feature_names))
        for feature, importance in xgb_importance_dict.items():
            # Предполагаем, что признаки в XGBoost имеют формат 'f0', 'f1', ... или соответствуют именам
            if feature.startswith('f') and feature[1:].isdigit():
                index = int(feature[1:])
                if index < len(xgb_importance):
                    xgb_importance[index] = importance
            elif feature in feature_names:
                index = feature_names.index(feature)
                xgb_importance[index] = importance
    else:  # для scikit-learn API (XGBRegressor, XGBClassifier)
        xgb_importance = model.feature_importances_
    
    # Создаем DataFrame для XGBoost importance
    xgb_importance_df = pd.DataFrame({
        'Feature': feature_names,
        f'XGB_{importance_type.capitalize()}_Importance': xgb_importance
    })
    
    # Получаем SHAP values
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        
        # Обрабатываем разные форматы SHAP values
        if isinstance(shap_values, list):  # для мультиклассовой классификации
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:  # для регрессии или бинарной классификации
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Создаем DataFrame для SHAP importance
        shap_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': shap_importance
        })
        
        # Объединяем оба DataFrame
        combined_importance = pd.merge(shap_importance_df, xgb_importance_df, on='Feature')
        
        # Сортируем по SHAP importance
        combined_importance = combined_importance.sort_values('SHAP_Importance', ascending=False)
        
        # Визуализация, если требуется
        if plot:
            plt.figure(figsize=figsize)
            
            # Создаем два подграфика
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
            
            # Строим график для SHAP importance
            combined_importance.sort_values('SHAP_Importance', ascending=True).plot.barh(
                x='Feature', y='SHAP_Importance', ax=ax1
            )
            ax1.set_title('SHAP Importance')
            
            # Строим график для XGBoost importance
            combined_importance.sort_values(f'XGB_{importance_type.capitalize()}_Importance', ascending=True).plot.barh(
                x='Feature', y=f'XGB_{importance_type.capitalize()}_Importance', ax=ax2
            )
            ax2.set_title(f'XGBoost {importance_type.capitalize()} Importance')
            
            plt.tight_layout()
            plt.show()
            
            # Дополнительно добавляем SHAP summary plot, если требуется visualization
            shap.summary_plot(shap_values, X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=feature_names), 
                             plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            plt.tight_layout()
            plt.show()
        
        if return_shap_values:
            return combined_importance, shap_values
        else:
            return combined_importance
            
    except Exception as e:
        print(f"Произошла ошибка при расчете SHAP values: {e}")
        # Если SHAP не удался, возвращаем только XGBoost importance
        xgb_importance_df = xgb_importance_df.sort_values(
            f'XGB_{importance_type.capitalize()}_Importance', ascending=False
        )
        return xgb_importance_df


def forecast_sales_prophet(time_series, periods=12, target='final_price'):
    """
    Прогнозирование с использованием Prophet
    """
    try:
        from prophet import Prophet
        from sklearn.preprocessing import LabelEncoder
        
        print(f"Running Prophet forecast for {target}...")
        
        # Создаем словарь для хранения энкодеров
        encoders = {}
        
        # Подготовка данных для Prophet
        df_prophet = pd.DataFrame({
            'ds': time_series.index,
            'y': time_series[target].values
        })
        
        # Создание модели Prophet с логистическим ростом для предотвращения отрицательных значений
        model = Prophet(
            growth='logistic',
            seasonality_mode='multiplicative',
            yearly_seasonality=True, 
            weekly_seasonality=False, 
            daily_seasonality=False
        )
        
        # Установка нижней и верхней границы для логистического роста
        df_prophet['floor'] = 0
        df_prophet['cap'] = df_prophet['y'].max() * 1.5
        
        # Обработка числовых и категориальных регрессоров
        for col in time_series.columns:
            # Пропускаем целевую переменную
            if col != target:
                # Преобразуем имя колонки в строку
                col_name = str(col)
                
                # Проверяем, является ли столбец числовым или категориальным
                if pd.api.types.is_numeric_dtype(time_series[col]):
                    # Для числовых столбцов добавляем как есть
                    df_prophet[col_name] = time_series[col].values
                    model.add_regressor(col_name)
                elif pd.api.types.is_categorical_dtype(time_series[col]) or pd.api.types.is_object_dtype(time_series[col]):
                    # Для категориальных создаем и сохраняем энкодер
                    encoders[col] = LabelEncoder()
                    # Преобразуем в строки перед кодированием для безопасности
                    encoded_values = encoders[col].fit_transform(time_series[col].astype(str))
                    df_prophet[col_name] = encoded_values
                    model.add_regressor(col_name)
        
        # Обучение модели
        model.fit(df_prophet)
        
        # Создание фрейма для прогноза
        future = model.make_future_dataframe(periods=periods, freq='MS')
        
        # Добавляем границы для логистического роста
        future['floor'] = 0
        future['cap'] = df_prophet['cap'].max()
        
        # Создаем копии закодированных признаков
        encoded_data = pd.DataFrame(index=time_series.index)
        
        # Для каждой колонки сохраняем оригинальные и закодированные значения
        for col in time_series.columns:
            if col != target:
                if col in encoders:
                    # Сохраняем закодированные значения
                    encoded_data[col] = df_prophet[col].values
                else:
                    # Сохраняем оригинальные значения для числовых признаков
                    encoded_data[col] = time_series[col].values
        
        # Объединяем с будущими датафреймами
        future = pd.merge(
            future, 
            encoded_data, 
            left_on='ds', 
            right_index=True, 
            how='left'
        )
        
        # Определяем, какие строки относятся к будущим датам (те, где будут NaN)
        is_future_date = future['ds'].apply(
            lambda x: x not in time_series.index
        )
        future_indices = future[is_future_date].index
        
        # Заполняем пропуски (будущие даты) последними известными значениями
        for col in encoded_data.columns:
            # Проверяем наличие NaN
            if future[col].isna().any():
                # Берем последнее известное значение
                last_value = encoded_data[col].iloc[-1]
                # Заполняем только для будущих дат
                future.loc[future_indices, col] = last_value
        
        # Проверка наличия NaN перед прогнозированием
        nan_columns = future.columns[future.isna().any()].tolist()
        if nan_columns:
            print(f"Обнаружены NaN в следующих столбцах Prophet: {nan_columns}")
            # Автоматическое заполнение пропущенных значений
            for col in nan_columns:
                # Используем среднее или моду в зависимости от типа данных
                if pd.api.types.is_numeric_dtype(future[col]):
                    fill_value = future[col].mean()
                else:
                    # Для нечисловых данных используем наиболее частое значение
                    non_na_values = future[col].dropna()
                    fill_value = non_na_values.value_counts().idxmax() if not non_na_values.empty else 0
                
                future[col] = future[col].fillna(fill_value)
        
        # Прогнозирование
        forecast_prophet = model.predict(future)
        
        # Создание датафрейма прогноза
        forecast_df = pd.DataFrame({
            f'forecast_{target}': forecast_prophet.tail(periods)['yhat'].values,
            f'{target}_lower': forecast_prophet.tail(periods)['yhat_lower'].values,
            f'{target}_upper': forecast_prophet.tail(periods)['yhat_upper'].values
        }, index=pd.date_range(start=time_series.index[-1] + pd.DateOffset(months=1), periods=periods, freq='MS'))
        
        # Применяем abs чтобы убедиться, что значения не отрицательные
        forecast_df[f'forecast_{target}'] = forecast_df[f'forecast_{target}'].abs()
        forecast_df[f'{target}_lower'] = forecast_df[f'{target}_lower'].abs()
        forecast_df[f'{target}_upper'] = forecast_df[f'{target}_upper'].abs()
        
        print(f"Prophet forecast complete for {target}")
        return forecast_df
        
    except Exception as e:
        print(f"Error in Prophet forecasting: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def forecast_sales_xgboost(time_series, periods=12, target='final_price'):
    """
    Прогнозирование с использованием XGBoost
    """
    try:
        import xgboost as xgb
        from sklearn.preprocessing import LabelEncoder
        
        print(f"Running XGBoost forecast for {target}...")
        
        # Создаем копию данных для безопасности
        data_copy = time_series.copy()
        
        # Применяем логарифмическое преобразование к целевым переменным
        # Добавляем 1, чтобы избежать log(0)
        data_copy[f'{target}_log'] = np.log1p(data_copy[target])
        
        # Кодирование категориальных переменных
        encoders = {}
        for col in data_copy.columns:
            if col not in [target, f'{target}_log']:
                if not pd.api.types.is_numeric_dtype(data_copy[col]):
                    encoders[col] = LabelEncoder()
                    data_copy[col] = encoders[col].fit_transform(data_copy[col].astype(str))
        
        # Функция для создания признаков
        def create_features(df, label=None, lags=None, ensure_features=None):
            df_new = df.copy()
            
            # Создание лагов
            if lags is None:
                lags = [1, 2, 3, 6, 12]
            
            # Используем логарифмированные значения для лагов
            for lag in lags:
                df_new[f'{label}_lag_{lag}'] = df_new[label].shift(lag)
            
            # Добавление временных признаков
            df_new['month'] = df_new.index.month
            df_new['quarter'] = df_new.index.quarter
            df_new['year'] = df_new.index.year
            
            # Добавление скользящего среднего для логарифмированных значений
            for window in [3, 6, 12]:
                df_new[f'{label}_rolling_{window}'] = df_new[label].rolling(window=window).mean()
            
            # Проверка наличия всех необходимых признаков
            if ensure_features is not None:
                for feature in ensure_features:
                    if feature not in df_new.columns and feature != label:
                        df_new[feature] = 0
            
            # Заполнение пропусков и получение признаков
            df_new = df_new.fillna(df_new.mean())
            all_features = [col for col in df_new.columns if col != label and col != target]
            
            if label and label in df_new.columns:
                X = df_new[all_features]
                y = df_new[label]
            else:
                X = df_new
                y = None
            
            return X, y, all_features
        
        # Создание признаков для обучения с логарифмированными целевыми переменными
        X_train, y_train, feature_list = create_features(data_copy, label=f'{target}_log')
        
        # Обучение модели XGBoost
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Рекурсивное прогнозирование
        forecast_log = []
        forecast_dates = []
        
        # Копируем последние данные для прогноза
        future_data = data_copy.copy()
        last_date = future_data.index[-1]
        
        # Итеративное прогнозирование
        for i in range(periods):
            next_date = last_date + pd.DateOffset(months=i+1)
            forecast_dates.append(next_date)
            
            # Создаем признаки для прогноза
            X_future, _, _ = create_features(future_data, label=f'{target}_log', ensure_features=feature_list)
            
            # Прогнозирование в логарифмической шкале
            log_pred = model.predict(X_future.iloc[-1:])
            forecast_log.append(log_pred[0])
            
            # Обновление данных - добавляем как обычные, так и логарифмированные значения
            new_row = pd.DataFrame({
                target: [np.expm1(log_pred[0])],
                f'{target}_log': [log_pred[0]]
            }, index=[next_date])
            
            # Копируем все остальные признаки из последней строки
            for col in future_data.columns:
                if col not in [target, f'{target}_log']:
                    new_row[col] = future_data[col].iloc[-1]
            
            # Добавляем строку
            future_data = pd.concat([future_data, new_row])
        
        # Обратное преобразование из логарифмической шкалы (exp(x)-1)
        forecast_values = np.expm1(forecast_log)
        forecast_values = np.abs(forecast_values)  # Убедимся, что все значения положительные
        
        # Создание датафрейма прогноза
        forecast_df = pd.DataFrame({
            f'forecast_{target}': forecast_values,
            f'{target}_lower': forecast_values * 0.9,  # Примерные доверительные интервалы
            f'{target}_upper': forecast_values * 1.1
        }, index=forecast_dates)
        
        print(f"XGBoost forecast complete for {target}")
        return forecast_df
        
    except Exception as e:
        print(f"Error in XGBoost forecasting: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def create_training_callback():
    """
    Создает callback функцию для мониторинга обучения AutoGluon
    """
    def training_callback(model_name, model_score, elapsed_time):
        """
        Callback функция для вывода метрик во время обучения
        """
        mape_score = abs(model_score)  # Преобразуем в положительное значение MAPE
        print(f"⏰ {elapsed_time:6.1f}s | 📊 Модель: {model_name:25s} | 🎯 MAPE: {mape_score:7.4f}")
    
    return training_callback


def forecast_sales_autogluon(time_series_df, periods=12, target='final_price', item_id_col='item_id', timestamp_col='date_key', models_dir='models', min_ts_length=None):
    """
    Прогнозирование продаж с использованием AutoGluon TimeSeries.
    
    Parameters:
    -----------
    time_series_df : pd.DataFrame
        Датафрейм с временными рядами
    periods : int
        Количество периодов для прогноза
    target : str
        Целевая переменная для прогноза
    item_id_col : str
        Название колонки с идентификатором товара
    timestamp_col : str
        Название колонки с временными метками
    models_dir : str
        Путь для сохранения моделей
    min_ts_length : int, optional
        Минимальная длина временного ряда (количество точек) для включения в прогнозирование
    """
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        import os
        import logging
        
        print(f"Running AutoGluon forecast for {target} with {periods} periods...")
        
        # Check required columns
        required_cols = [item_id_col, timestamp_col, target]
        if not all(col in time_series_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in time_series_df.columns]
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Ensure timestamp is datetime
        time_series_df[timestamp_col] = pd.to_datetime(time_series_df[timestamp_col])
        
        # Ensure target is numeric
        time_series_df[target] = pd.to_numeric(time_series_df[target], errors='coerce').fillna(0)
        
        # Convert to TimeSeriesDataFrame
        ts_data = TimeSeriesDataFrame.from_data_frame(
            time_series_df,
            id_column=item_id_col,
            timestamp_column=timestamp_col
        )
        
        print(f"Prepared AutoGluon data with {ts_data.num_items} time series")
        
        # Фильтрация коротких временных рядов если задан min_ts_length
        if min_ts_length is not None and min_ts_length > 0:
            # Считаем количество точек для каждого ряда
            ts_lengths = ts_data.num_timesteps_per_item().to_dict()
            
            # Находим ряды с достаточным количеством точек
            valid_items = [item_id for item_id, length in ts_lengths.items() if length >= min_ts_length]
            
            # Фильтруем данные
            if valid_items:
                print(f"Filtering time series: {ts_data.num_items} → {len(valid_items)} items " +
                      f"(removed {ts_data.num_items - len(valid_items)} with < {min_ts_length} points)")
                ts_data = ts_data.loc[valid_items]
            else:
                print(f"Warning: No time series with >= {min_ts_length} points found!")
        
        # Регуляризация временных рядов с ежемесячной частотой (MS - month start)
        try:
            print("Converting time series to regular monthly frequency...")
            ts_data = ts_data.convert_frequency(freq="MS")
            print("Time series frequency conversion completed")
        except Exception as e:
            print(f"Warning: Could not convert frequency: {e}")
            print("Will proceed with explicit frequency setting")
        
        # Создаем путь для сохранения моделей
        model_path = os.path.join(models_dir, f"autogluon_model_{target}")
        os.makedirs(models_dir, exist_ok=True)
        
        # Разделяем данные на train/validation для получения метрик валидации
        # Берем последние периоды для валидации
        validation_length = min(6, periods)  # 6 месяцев или меньше для валидации
        
        print(f"Настройка валидации с {validation_length} периодами...")
        
        # Initialize and train predictor 
        predictor = TimeSeriesPredictor(
            prediction_length=periods,
            path=model_path,
            target=target,
            eval_metric="MAPE",
            freq="MS",  # Explicitly set monthly frequency (month start)
            verbosity=2  # Умеренный уровень подробности (вместо 4)
        )
        
        print("Начинаем обучение AutoGluon...")
        print("🔄 Будут показаны MAPE метрики для каждой обученной модели")
        print("=" * 80)
        
        # Обучение с умеренным выводом информации
        predictor.fit(
            ts_data,
            presets="best_quality",
            time_limit=3600,  
            verbosity=2,  # Умеренный уровень подробности
            # Исключаем проблемные модели
            excluded_model_types=['DynamicOptimizedTheta']
        )
        
        print("=" * 80)
        print("AutoGluon training complete")
        
        # Выводим только итоговые метрики
        try:
            leaderboard = predictor.leaderboard(silent=True)
            print("\n📊 ИТОГОВЫЕ МЕТРИКИ ВАЛИДАЦИИ (MAPE):")
            print("=" * 60)
            
            for idx, (model_name, row) in enumerate(leaderboard.iterrows()):
                mape_val = abs(row['score_val'])  # Преобразуем в положительное значение
                model_name_short = str(model_name)[:30]  # Ограничиваем длину имени
                training_time = row.get('fit_time_marginal', 0)
                
                print(f"{idx+1:2d}. {model_name_short:30s} | MAPE: {mape_val:7.4f} | Время: {training_time:6.1f}s")
            
            best_mape = abs(leaderboard.iloc[0]['score_val'])
            best_model = str(leaderboard.index[0])
            print("=" * 60)
            print(f"🏆 Лучшая модель: {best_model}")
            print(f"🎯 Лучшая MAPE: {best_mape:.4f}")
            print("=" * 60)
            
        except Exception as e:
            print(f"Не удалось получить метрики валидации: {e}")
        
        # Make predictions
        print(f"\nВыполняем прогнозирование для {target}...")
        predictions = predictor.predict(ts_data)
        print(f"AutoGluon prediction complete for {target}")
        
        # Return predictions with reset index for easier handling
        return predictions.reset_index()
        
    except Exception as e:
        print(f"Error in AutoGluon forecasting: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def forecast_sales_average(time_series, periods=12, target='final_price'):
    """
    Простое прогнозирование методом среднего значения
    """
    try:
        # Используем среднее за последние периоды
        window = min(6, len(time_series))
        avg_value = time_series[target].tail(window).mean()
        
        # Создание датафрейма прогноза
        forecast_dates = pd.date_range(
            start=time_series.index[-1] + pd.DateOffset(months=1),
            periods=periods,
            freq='MS'
        )
        
        forecast_df = pd.DataFrame({
            f'forecast_{target}': [avg_value] * periods,
            f'{target}_lower': [avg_value * 0.8] * periods,  # примерный нижний интервал
            f'{target}_upper': [avg_value * 1.2] * periods   # примерный верхний интервал
        }, index=forecast_dates)
        
        return forecast_df
        
    except Exception as e:
        print(f"Error in average forecasting: {e}")
        return pd.DataFrame()


def forecast_sales(time_series, periods=12, method='prophet', target='final_price', **kwargs):
    """
    Унифицированный интерфейс для различных методов прогнозирования
    """
    if time_series.empty:
        print("Empty time series. Forecasting not possible.")
        return pd.DataFrame()
    
    if method == 'prophet':
        return forecast_sales_prophet(time_series, periods, target)
    elif method == 'xgboost':
        return forecast_sales_xgboost(time_series, periods, target)
    elif method == 'autogluon':
        # For autogluon, we expect time_series to have item_id and timestamp columns
        # This is different from other methods which expect a DatetimeIndex
        item_id_col = kwargs.get('item_id_col', 'item_id')
        timestamp_col = kwargs.get('timestamp_col', 'date_key')
        models_dir = kwargs.get('models_dir', 'models')
        min_ts_length = kwargs.get('min_ts_length', None)
        return forecast_sales_autogluon(
            time_series, periods, target, item_id_col, 
            timestamp_col, models_dir, min_ts_length
        )
    elif method == 'average':
        return forecast_sales_average(time_series, periods, target)
    else:
        print(f"Method '{method}' not supported.")
        print("Supported methods: 'prophet', 'xgboost', 'autogluon', 'average'")
        return pd.DataFrame()


def forecast_by_segment(data, segment_column, periods=12, method="xgboost", target='final_price'):
    """
    Прогнозирование продаж по сегментам (категория, группа, регион и т.д.)
    """
    from data_processing import prepare_time_series
    
    # Проверка наличия колонки сегмента
    if segment_column not in data.columns:
        raise ValueError(f"Column '{segment_column}' not found in data")
    
    # Получение уникальных значений сегмента
    segments = data[segment_column].unique()
    
    # Словарь для хранения прогнозов
    forecasts = {}
    
    # Прогнозирование для каждого сегмента
    for segment in segments:
        segment_data = data[data[segment_column] == segment]
        
        # Пропускаем сегменты с недостаточным количеством данных
        if len(segment_data) < 12:
            continue
        
        # Подготовка временного ряда
        ts = prepare_time_series(segment_data)
        
        # Прогнозирование
        forecast = forecast_sales(ts, method=method, periods=periods, target=target)
        
        # Сохранение результата
        forecasts[segment] = forecast
    
    return forecasts


def analyze_autogluon_validation(predictor, save_path=None):
    """
    Детальный анализ результатов валидации AutoGluon TimeSeries
    
    Parameters:
    -----------
    predictor : TimeSeriesPredictor
        Обученный предиктор AutoGluon
    save_path : str, optional
        Путь для сохранения результатов анализа
    
    Returns:
    --------
    dict : Словарь с результатами анализа
    """
    try:
        print("=" * 80)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ ВАЛИДАЦИИ AUTOGLUON")
        print("=" * 80)
        
        results = {}
        
        # 1. Получаем leaderboard
        leaderboard = predictor.leaderboard(silent=True)
        results['leaderboard'] = leaderboard
        
        print("\n1. ТАБЛИЦА ЛИДЕРОВ (отсортирована по MAPE):")
        print("-" * 50)
        print(leaderboard.to_string())
        
        # 2. Лучшая модель
        if len(leaderboard) == 0:
            print("\n❌ Leaderboard пуст - модели не были обучены или произошла ошибка")
            return {}
            
        best_model_name = leaderboard.index[0]
        best_score = leaderboard.iloc[0]['score_val']
        
        # AutoGluon возвращает отрицательные значения для MAPE (чем меньше, тем лучше)
        # Преобразуем в положительные для удобства
        best_mape = abs(best_score)
        
        results['best_model'] = str(best_model_name)  # Преобразуем в строку
        results['best_mape'] = best_mape
        
        print(f"\n2. ЛУЧШАЯ МОДЕЛЬ:")
        print("-" * 50)
        print(f"Модель: {best_model_name}")
        print(f"MAPE на валидации: {best_mape:.4f}")
        
        # 3. Сравнение всех моделей
        print(f"\n3. СРАВНЕНИЕ МОДЕЛЕЙ ПО MAPE:")
        print("-" * 50)
        for idx, (model_name, row) in enumerate(leaderboard.iterrows()):
            mape_val = abs(row['score_val'])  # Преобразуем в положительное значение
            model_name_str = str(model_name)[:25]  # Ограничиваем длину и преобразуем в строку
            print(f"{idx+1:2d}. {model_name_str:25s} | MAPE: {mape_val:7.4f}")
        
        # 4. Статистика по моделям
        mape_scores = abs(leaderboard['score_val'].values)  # Преобразуем в положительные значения
        results['mape_stats'] = {
            'mean': np.mean(mape_scores),
            'std': np.std(mape_scores),
            'min': np.min(mape_scores),
            'max': np.max(mape_scores),
            'median': np.median(mape_scores)
        }
        
        print(f"\n4. СТАТИСТИКА MAPE ПО ВСЕМ МОДЕЛЯМ:")
        print("-" * 50)
        print(f"Среднее MAPE:     {results['mape_stats']['mean']:.4f}")
        print(f"Стд. отклонение:  {results['mape_stats']['std']:.4f}")
        print(f"Минимальное MAPE: {results['mape_stats']['min']:.4f}")
        print(f"Максимальное MAPE:{results['mape_stats']['max']:.4f}")
        print(f"Медианное MAPE:   {results['mape_stats']['median']:.4f}")
        
        # 5. Попытка получить дополнительную информацию о моделях
        try:
            model_info = {}
            for model_name in leaderboard.index[:5]:  # Топ-5 моделей
                try:
                    model = predictor._trainer.load_model(model_name)
                    if hasattr(model, 'get_info'):
                        model_info[model_name] = model.get_info()
                except:
                    continue
            
            if model_info:
                results['model_info'] = model_info
                print(f"\n5. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ О МОДЕЛЯХ:")
                print("-" * 50)
                for model_name, info in model_info.items():
                    print(f"{model_name}: {info}")
        except:
            pass
        
        # 6. Сохранение результатов
        if save_path:
            try:
                import json
                import os
                
                # Создаем директорию если не существует
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # Сохраняем результаты в JSON
                results_to_save = {
                    'best_model': results['best_model'],
                    'best_mape': float(results['best_mape']),
                    'mape_stats': {k: float(v) for k, v in results['mape_stats'].items()},
                    'leaderboard': results['leaderboard'].to_dict()
                }
                
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(results_to_save, f, ensure_ascii=False, indent=2)
                
                print(f"\n6. Результаты сохранены в: {save_path}")
                
                # Также сохраняем leaderboard в CSV
                csv_path = save_path.replace('.json', '_leaderboard.csv')
                results['leaderboard'].to_csv(csv_path)
                print(f"   Leaderboard сохранен в: {csv_path}")
                
            except Exception as e:
                print(f"Ошибка при сохранении: {e}")
        
        print("=" * 80)
        return results
        
    except Exception as e:
        print(f"Ошибка при анализе результатов валидации: {e}")
        import traceback
        traceback.print_exc()
        return {}


def load_and_analyze_autogluon_model(model_path, save_validation_results=True):
    """
    Загружает обученную модель AutoGluon и анализирует ее результаты валидации
    
    Parameters:
    -----------
    model_path : str
        Путь к папке с сохраненной моделью AutoGluon
    save_validation_results : bool
        Сохранять ли результаты анализа в файл
    
    Returns:
    --------
    dict : Словарь с результатами анализа или None если модель не найдена
    """
    try:
        from autogluon.timeseries import TimeSeriesPredictor
        import os
        
        if not os.path.exists(model_path):
            print(f"Модель не найдена по пути: {model_path}")
            return None
        
        print(f"Загружаем модель AutoGluon из: {model_path}")
        
        # Загружаем предиктор
        predictor = TimeSeriesPredictor.load(model_path)
        
        print("Модель успешно загружена!")
        print(f"Целевая переменная: {predictor.target}")
        print(f"Длина прогноза: {predictor.prediction_length}")
        print(f"Частота данных: {predictor.freq}")
        
        # Выполняем анализ валидации
        if save_validation_results:
            results_path = os.path.join(model_path, "validation_analysis.json")
        else:
            results_path = None
            
        validation_results = analyze_autogluon_validation(predictor, results_path)
        
        return {
            'predictor': predictor,
            'validation_results': validation_results,
            'model_info': {
                'target': predictor.target,
                'prediction_length': predictor.prediction_length,
                'freq': predictor.freq,
                'path': model_path
            }
        }
        
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_autogluon_models(model_paths, target_names=None):
    """
    Сравнивает несколько моделей AutoGluon по метрикам валидации
    
    Parameters:
    -----------
    model_paths : list
        Список путей к моделям AutoGluon
    target_names : list, optional
        Названия целевых переменных для отображения
    
    Returns:
    --------
    pd.DataFrame : Таблица сравнения моделей
    """
    try:
        print("=" * 80)
        print("СРАВНЕНИЕ МОДЕЛЕЙ AUTOGLUON")
        print("=" * 80)
        
        comparison_results = []
        
        for i, model_path in enumerate(model_paths):
            try:
                model_info = load_and_analyze_autogluon_model(model_path, save_validation_results=False)
                
                if model_info is None:
                    continue
                
                target_name = target_names[i] if target_names and i < len(target_names) else f"Model_{i+1}"
                validation_results = model_info['validation_results']
                
                if 'best_mape' in validation_results:
                    comparison_results.append({
                        'Модель': target_name,
                        'Целевая переменная': model_info['model_info']['target'],
                        'Лучшая MAPE': validation_results['best_mape'],
                        'Средняя MAPE': validation_results['mape_stats']['mean'],
                        'Мин MAPE': validation_results['mape_stats']['min'],
                        'Макс MAPE': validation_results['mape_stats']['max'],
                        'Количество моделей': len(validation_results['leaderboard']),
                        'Путь': model_path
                    })
                    
            except Exception as e:
                print(f"Ошибка при обработке модели {model_path}: {e}")
                continue
        
        if not comparison_results:
            print("Не удалось загрузить ни одну модель для сравнения")
            return pd.DataFrame()
        
        # Создаем DataFrame с результатами
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df = comparison_df.sort_values('Лучшая MAPE')
        
        print("\nСРАВНЕНИЕ МОДЕЛЕЙ ПО MAPE:")
        print("-" * 80)
        print(comparison_df.to_string(index=False))
        
        # Определяем лучшую модель
        best_model = comparison_df.iloc[0]
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ:")
        print(f"   Название: {best_model['Модель']}")
        print(f"   Целевая переменная: {best_model['Целевая переменная']}")
        print(f"   MAPE: {best_model['Лучшая MAPE']:.4f}")
        print(f"   Путь: {best_model['Путь']}")
        
        return comparison_df
        
    except Exception as e:
        print(f"Ошибка при сравнении моделей: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def get_autogluon_model_summary(models_dir='models'):
    """
    Получает сводку по всем моделям AutoGluon в указанной директории
    
    Parameters:
    -----------
    models_dir : str
        Путь к директории с моделями
    
    Returns:
    --------
    dict : Сводная информация по моделям
    """
    try:
        import os
        import glob
        
        print(f"Поиск моделей AutoGluon в директории: {models_dir}")
        
        # Ищем все поддиректории с моделями AutoGluon
        model_paths = glob.glob(os.path.join(models_dir, "autogluon_model_*"))
        
        if not model_paths:
            print("Модели AutoGluon не найдены")
            return {}
        
        print(f"Найдено {len(model_paths)} моделей AutoGluon:")
        for path in model_paths:
            print(f"  - {os.path.basename(path)}")
        
        # Извлекаем названия целевых переменных из путей
        target_names = [os.path.basename(path).replace('autogluon_model_', '') for path in model_paths]
        
        # Сравниваем модели
        comparison_df = compare_autogluon_models(model_paths, target_names)
        
        summary = {
            'total_models': len(model_paths),
            'model_paths': model_paths,
            'target_names': target_names,
            'comparison_results': comparison_df
        }
        
        return summary
        
    except Exception as e:
        print(f"Ошибка при получении сводки моделей: {e}")
        import traceback
        traceback.print_exc()
        return {} 