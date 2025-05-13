import os
import sys
import subprocess
import pandas as pd
import numpy as np
from pathlib import Path

def check_file_exists(file_path):
    """Проверяет существование файла"""
    return os.path.exists(file_path)

def run_analysis():
    """Запускает анализ данных"""
    print("\n" + "=" * 80)
    print("ЗАПУСК АНАЛИЗА ДАННЫХ".center(80))
    print("=" * 80)
    
    # Путь к скрипту main.py
    project_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(project_dir, 'scripts', 'main.py')
    
    if not os.path.exists(script_path):
        print(f"Ошибка: скрипт {script_path} не найден!")
        return False
    
    try:
        print(f"Запуск скрипта {script_path}")
        subprocess.run([sys.executable, script_path], check=True)
        print("Анализ успешно завершен")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при выполнении скрипта: {e}")
        return False

def check_dependencies():
    """Проверяет наличие необходимых зависимостей"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ЗАВИСИМОСТЕЙ".center(80))
    print("=" * 80)
    
    missing_packages = []
    
    # Список необходимых пакетов
    required_packages = [
        "pandas",
        "numpy",
        "matplotlib",
        "pyarrow"  # для поддержки Parquet
    ]
    
    # Проверка наличия необходимых пакетов
    for package in required_packages:
        try:
            __import__(package)
            print(f"Пакет {package} установлен")
        except ImportError:
            missing_packages.append(package)
            print(f"Пакет {package} не установлен")
    
    # Установка недостающих пакетов
    if missing_packages:
        print("\nУстановка недостающих пакетов...")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"Пакет {package} успешно установлен")
            except subprocess.CalledProcessError:
                print(f"Не удалось установить пакет {package}")
                return False
    
    # Проверка возможности работы с Parquet
    try:
        pd.DataFrame().to_parquet
        print("Поддержка Parquet: OK")
    except AttributeError:
        print("Ошибка: поддержка Parquet отсутствует")
        return False
    
    print("Все необходимые зависимости установлены\n")
    return True

def check_data_files():
    """Проверяет наличие необходимых файлов данных"""
    print("\n" + "=" * 80)
    print("ПРОВЕРКА ФАЙЛОВ ДАННЫХ".center(80))
    print("=" * 80)
    
    # Получаем текущую директорию проекта
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, 'data')
    
    # Создаем директорию для данных, если нужно
    os.makedirs(data_dir, exist_ok=True)
    
    # Список файлов для проверки
    required_files = [
        os.path.join(data_dir, 'final_df_latest2.parquet'),
        os.path.join(data_dir, 'denied.parquet')
    ]
    
    # Проверяем каждый файл
    missing_files = []
    for file_path in required_files:
        if not check_file_exists(file_path):
            missing_files.append(file_path)
            print(f"Файл не найден: {file_path}")
        else:
            print(f"Файл найден: {file_path}")
    
    if missing_files:
        print(f"\nОтсутствуют необходимые файлы данных: {len(missing_files)}")
        return False
    
    print("Все необходимые файлы данных найдены")
    return True

def main():
    """Основная функция"""
    print("=" * 80)
    print("ЗАПУСК ПОЛНОГО АНАЛИЗА ПРОДАЖ".center(80))
    print("=" * 80)
    
    print(f"Текущая директория: {os.getcwd()}")
    
    # Шаг 1: Проверка зависимостей
    if not check_dependencies():
        print("Ошибка: проверка зависимостей не пройдена")
        return
    
    # Шаг 2: Проверка наличия файлов данных
    if not check_data_files():
        print("Ошибка: отсутствуют необходимые файлы данных")
        return
    
    # Шаг 3: Запуск анализа
    if not run_analysis():
        print("Ошибка: анализ не был выполнен")
        return
    
    print("\n" + "=" * 80)
    print("ПОЛНЫЙ АНАЛИЗ УСПЕШНО ЗАВЕРШЕН".center(80))
    print("=" * 80)
    
    # Выводим информацию о созданных файлах
    project_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(project_dir, 'results')
    
    if os.path.exists(results_dir):
        result_files = [f for f in os.listdir(results_dir) if f.endswith('.parquet')]
        if result_files:
            print("\nСозданные файлы результатов:")
            for f in result_files:
                print(f" - {f}")
    
    print("\nВсе файлы сохранены в формате Parquet для эффективного использования!")

if __name__ == "__main__":
    main() 