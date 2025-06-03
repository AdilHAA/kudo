#!/usr/bin/env python3
"""
Тестовый скрипт для проверки вывода метрик MAPE во время обучения AutoGluon
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Добавляем путь к директории src для импорта модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import forecast_sales_autogluon


def create_test_data():
    """
    Создает тестовые данные для демонстрации
    """
    print("Создаем тестовые данные...")
    
    # Создаем данные для 3 SKU за 24 месяца
    dates = pd.date_range(start='2022-01-01', periods=24, freq='MS')
    
    data = []
    for item_id in ['SKU_001', 'SKU_002', 'SKU_003']:
        for date in dates:
            # Добавляем тренд и сезонность
            trend = np.random.normal(100, 10)
            seasonal = 20 * np.sin(2 * np.pi * date.month / 12)
            noise = np.random.normal(0, 5)
            
            final_price = max(0, trend + seasonal + noise)
            quantity = max(1, np.random.poisson(10) + int(seasonal/2))
            
            data.append({
                'item_id': item_id,
                'date_key': date,
                'final_price': final_price,
                'quantity': quantity
            })
    
    df = pd.DataFrame(data)
    print(f"Создано {len(df)} записей для {df['item_id'].nunique()} SKU")
    return df


def test_training_with_metrics():
    """
    Тестирует обучение AutoGluon с выводом метрик во время обучения
    """
    print("=" * 80)
    print("ТЕСТ ВЫВОДА МЕТРИК ВО ВРЕМЯ ОБУЧЕНИЯ AUTOGLUON")
    print("=" * 80)
    
    # Создаем тестовые данные
    test_data = create_test_data()
    
    # Создаем директорию для тестовых моделей
    test_models_dir = '../models/test_models'
    os.makedirs(test_models_dir, exist_ok=True)
    
    print(f"\n🎯 Начинаем тестовое обучение для переменной 'final_price'")
    print("💡 Обратите внимание на вывод метрик MAPE во время обучения каждой модели")
    print("-" * 80)
    
    try:
        # Запускаем обучение с выводом метрик
        forecast_result = forecast_sales_autogluon(
            test_data,
            periods=6,  # Прогнозируем на 6 месяцев
            target='final_price',
            item_id_col='item_id',
            timestamp_col='date_key',
            models_dir=test_models_dir
        )
        
        if not forecast_result.empty:
            print("\n✅ ТЕСТОВОЕ ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО!")
            print(f"📊 Получен прогноз размером: {forecast_result.shape}")
            print("\nПервые 5 строк прогноза:")
            print(forecast_result.head())
        else:
            print("\n❌ Тестовое обучение не дало результатов")
            
    except Exception as e:
        print(f"\n❌ Ошибка во время тестового обучения: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("ТЕСТ ЗАВЕРШЕН")
    print("=" * 80)


if __name__ == "__main__":
    test_training_with_metrics() 