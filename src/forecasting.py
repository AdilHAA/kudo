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


def forecast_sales_autogluon(time_series_df, periods=12, target='final_price', item_id_col='item_id', timestamp_col='date_key', models_dir='models'):
    """
    Прогнозирование продаж с использованием AutoGluon TimeSeries.
    """
    try:
        from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
        
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
        
        # Initialize and train predictor with explicit frequency
        predictor = TimeSeriesPredictor(
            prediction_length=periods,
            path=model_path,
            target=target,
            eval_metric="MAPE",
            freq="MS"  # Explicitly set monthly frequency (month start)
        )
        
        predictor.fit(
            ts_data,
            presets="high_quality",
            time_limit=1200  
        )
        
        print("AutoGluon training complete")
        
        # Make predictions
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
        return forecast_sales_autogluon(time_series, periods, target, item_id_col, timestamp_col, models_dir)
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