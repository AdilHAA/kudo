#!/usr/bin/env python3
"""
Простой скрипт для анализа результатов валидации AutoGluon
"""

import os
import sys
import pandas as pd

# Добавляем путь к директории src для импорта модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.forecasting import (
    analyze_autogluon_validation,
    load_and_analyze_autogluon_model
)


def quick_analysis():
    """
    Быстрый анализ существующих моделей
    """
    print("="*60)
    print("АНАЛИЗ ВАЛИДАЦИИ AUTOGLUON МОДЕЛЕЙ")
    print("="*60)
    
    models_dir = '../models'
    model_paths = [
        os.path.join(models_dir, 'autogluon_model_final_price'),
        os.path.join(models_dir, 'autogluon_model_quantity')
    ]
    
    results = []
    
    for model_path in model_paths:
        if os.path.exists(model_path):
            target = os.path.basename(model_path).replace('autogluon_model_', '')
            print(f"\nАнализ модели для: {target}")
            print("-" * 40)
            
            try:
                model_info = load_and_analyze_autogluon_model(model_path, save_validation_results=True)
                
                if model_info and 'validation_results' in model_info:
                    val_results = model_info['validation_results']
                    
                    if 'best_mape' in val_results:
                        results.append({
                            'Целевая переменная': target,
                            'Лучшая MAPE': val_results['best_mape'],
                            'Средняя MAPE': val_results['mape_stats']['mean'],
                            'Мин MAPE': val_results['mape_stats']['min'],
                            'Макс MAPE': val_results['mape_stats']['max']
                        })
                        print(f"✅ Успешно проанализирована")
                        print(f"   Лучшая MAPE: {val_results['best_mape']:.4f}")
                    else:
                        print("❌ Не удалось получить метрики MAPE")
                else:
                    print("❌ Ошибка при загрузке модели")
                    
            except Exception as e:
                print(f"❌ Ошибка: {e}")
        else:
            print(f"❌ Модель не найдена: {model_path}")
    
    # Итоговое сравнение
    if results:
        print("\n" + "="*60)
        print("ИТОГОВОЕ СРАВНЕНИЕ")
        print("="*60)
        
        df = pd.DataFrame(results)
        df = df.sort_values('Лучшая MAPE')
        
        print("\nРезультаты (отсортированы по лучшей MAPE):")
        print(df.to_string(index=False))
        
        best_model = df.iloc[0]
        print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ:")
        print(f"   Переменная: {best_model['Целевая переменная']}")
        print(f"   MAPE: {best_model['Лучшая MAPE']:.4f}")
        
        # Сохраняем результаты
        results_path = os.path.join(models_dir, "validation_comparison.csv")
        df.to_csv(results_path, index=False)
        print(f"\n💾 Результаты сохранены в: {results_path}")
    else:
        print("\n❌ Не удалось проанализировать ни одну модель")


if __name__ == "__main__":
    quick_analysis() 