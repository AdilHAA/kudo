#!/usr/bin/env python3
"""
Скрипт для анализа обученных моделей AutoGluon
Выводит детальные метрики валидации, включая MAPE
"""

import os
import sys

# Добавляем путь к src для импорта модулей
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from forecasting import (
    get_autogluon_model_summary,
    load_and_analyze_autogluon_model,
    compare_autogluon_models
)


def main():
    """
    Основная функция для анализа моделей
    """
    print("=" * 80)
    print("АНАЛИЗ ОБУЧЕННЫХ МОДЕЛЕЙ AUTOGLUON")
    print("=" * 80)
    
    models_dir = 'models'
    
    if not os.path.exists(models_dir):
        print(f"Директория с моделями не найдена: {models_dir}")
        return
    
    # Получаем сводку по всем моделям
    print("1. Получение сводки по всем моделям...")
    summary = get_autogluon_model_summary(models_dir)
    
    if not summary:
        print("Модели не найдены для анализа")
        return
    
    print(f"\nВсего найдено моделей: {summary['total_models']}")
    
    # Детальный анализ каждой модели
    print("\n" + "=" * 80)
    print("2. ДЕТАЛЬНЫЙ АНАЛИЗ КАЖДОЙ МОДЕЛИ")
    print("=" * 80)
    
    for i, (model_path, target_name) in enumerate(zip(summary['model_paths'], summary['target_names'])):
        print(f"\n[{i+1}/{len(summary['model_paths'])}] Анализ модели для переменной: {target_name}")
        print("-" * 60)
        
        model_info = load_and_analyze_autogluon_model(model_path, save_validation_results=True)
        
        if model_info:
            print(f"✅ Модель успешно проанализирована")
            print(f"   Лучшая MAPE: {model_info['validation_results'].get('best_mape', 'N/A'):.4f}")
        else:
            print(f"❌ Ошибка при анализе модели")
    
    # Итоговое сравнение
    if 'comparison_results' in summary and not summary['comparison_results'].empty:
        print("\n" + "=" * 80)
        print("3. ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
        print("=" * 80)
        
        comparison_df = summary['comparison_results']
        
        print(f"\n🎯 ЛУЧШАЯ МОДЕЛЬ:")
        best_model = comparison_df.iloc[0]
        print(f"   Переменная: {best_model['Целевая переменная']}")
        print(f"   MAPE: {best_model['Лучшая MAPE']:.4f}")
        
        print(f"\n📊 СТАТИСТИКА ПО ВСЕМ МОДЕЛЯМ:")
        all_mapes = comparison_df['Лучшая MAPE']
        print(f"   Среднее MAPE: {all_mapes.mean():.4f}")
        print(f"   Лучшее MAPE:  {all_mapes.min():.4f}")
        print(f"   Худшее MAPE:  {all_mapes.max():.4f}")
        
        # Сохраняем результаты сравнения
        results_path = os.path.join(models_dir, "models_comparison.csv")
        comparison_df.to_csv(results_path, index=False)
        print(f"\n💾 Результаты сравнения сохранены в: {results_path}")


if __name__ == "__main__":
    main() 