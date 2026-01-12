# Файл: retro.py (для запуска из командной строки с AiZynthFinder)

# --- Импорты ---
import pandas as pd
import numpy as np
import time
import warnings
import os
import argparse
import traceback
import subprocess # Для запуска aizynthcli
import json
import tempfile
from tqdm import tqdm # Для прогресс-бара
from rdkit import Chem # Для проверки SMILES

# --- 1. НАСТРОЙКИ ---
# --- ПУТЬ К КОНФИГУРАЦИИ AiZynthFinder (ЗАДАТЬ ПРАВИЛЬНО!) ---
AIZYNTH_CONFIG_PATH = os.path.abspath('./aizynth_data/config.yml')

# --- Команда для запуска CLI (если окружение активировано) ---
AIZYNTH_COMMAND = ["aizynthcli"]
# --- Если нужно указать путь к окружению (раскомментировать и исправить): ---
# AIZYNTH_ENV_PATH = '/path/to/your/miniconda3/envs/aizynth-env'
# AIZYNTH_COMMAND = [os.path.join(AIZYNTH_ENV_PATH, 'bin', 'aizynthcli')]

# Имена колонок
DEFAULT_SMILES_COL = 'SMILES'
DEFAULT_INDEX_COL = 'original_index' # Опционально
DEFAULT_OUTPUT_FILENAME = 'retrosynthesis_aizynth_results.csv'

# Параметры AiZynthFinder
AIZYNTH_TIMEOUT = 300 # Таймаут на одну молекулу в секундах (5 минут)
AIZYNTH_NPROC = 1 # Количество процессов для ОДНОГО запуска aizynthcli

# --- Подавление предупреждений ---
warnings.filterwarnings("ignore")
from rdkit import rdBase
rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')

# --- 2. Функции ---
def count_reactions_in_tree(tree: dict) -> int:
    """Рекурсивно считает количество реакций (узлов с children) в дереве."""
    count = 0
    if 'children' in tree and isinstance(tree['children'], list) and len(tree['children']):
        count += 1
        for child_node in tree['children']:
            count += count_reactions_in_tree(child_node)
    return count

def run_single_retrosynthesis_aizynth(smiles: str):
    """
    Вызывает aizynthcli для одного SMILES (передавая его через файл) и парсит результат.
    Более устойчив к формату JSON.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {'synth_possible': False, 'synth_steps': np.nan, 'synth_score': np.nan, 'error': 'Invalid/Unparsable SMILES'}

    result_data = {'synth_possible': False, 'synth_steps': np.nan, 'synth_score': np.nan, 'error': None}
    temp_smiles_filename = None
    temp_output_json = tempfile.mktemp(suffix=".json")
    json_content_for_debug = None # Для отладки

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.smi', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(smiles + '\n')
            temp_smiles_filename = tmp_file.name

        if not temp_smiles_filename or not os.path.exists(temp_smiles_filename):
             raise IOError("Failed to create temporary SMILES file.")

        command = AIZYNTH_COMMAND + [
            "--config", AIZYNTH_CONFIG_PATH,
            "--smiles", temp_smiles_filename,
            "--output", temp_output_json,
            "--nproc", str(AIZYNTH_NPROC)
        ]

        process = subprocess.run(command, capture_output=True, text=True, timeout=AIZYNTH_TIMEOUT, check=False, encoding='utf-8')

        if process.returncode != 0:
            # ... (обработка ошибки CLI как раньше) ...
            error_message = process.stderr.strip() if process.stderr else "Unknown CLI error"
            error_lines = [line for line in error_message.split('\n') if not line.startswith(('W','E','I','202')) and 'factory' not in line and 'placer' not in line]
            clean_error = "\n".join(error_lines[-5:])
            result_data['error'] = f"AiZynthCLI Error (code {process.returncode}): {clean_error or error_message[:200]}"

        elif os.path.exists(temp_output_json):
            try:
                with open(temp_output_json, 'r', encoding='utf-8') as f:
                    json_text = f.read()
                    if not json_text.strip():
                         result_data['error'] = "Empty JSON output file"
                    else:
                        # --- ИЗМЕНЕНИЯ ЗДЕСЬ ---
                        output_data_parsed = json.loads(json_text) # Парсим JSON
                        json_content_for_debug = output_data_parsed # Сохраняем для отладки

                        output_data = None
                        # Проверяем, это список или словарь
                        if isinstance(output_data_parsed, list):
                             if len(output_data_parsed) > 0:
                                 output_data = output_data_parsed[0] # Берем первый элемент, если список не пуст
                             else:
                                 result_data['error'] = "JSON output is an empty list" # Список пуст
                        elif isinstance(output_data_parsed, dict):
                             output_data = output_data_parsed # Используем словарь напрямую
                        else:
                             result_data['error'] = "Unexpected JSON format (not list or dict)"

                        # Если удалось получить словарь output_data, парсим его
                        if output_data and isinstance(output_data, dict):
                            if output_data.get('status') == 'solved':
                                result_data['synth_possible'] = True
                                if 'trees' in output_data and output_data['trees'] and isinstance(output_data['trees'], list):
                                     try:
                                         # ... (парсинг деревьев как раньше) ...
                                         output_data['trees'].sort(key=lambda t: t.get('score', 0) if isinstance(t, dict) else 0, reverse=True)
                                         best_tree = output_data['trees'][0]
                                         if isinstance(best_tree, dict):
                                             depth = best_tree.get('metadata', {}).get('depth', best_tree.get('depth'))
                                             if depth is not None: result_data['synth_steps'] = int(depth)
                                             score = best_tree.get('score')
                                             if score is not None: result_data['synth_score'] = float(score)
                                         else: result_data['error'] = "Best tree format unexpected"
                                     except (KeyError, IndexError, TypeError, ValueError) as tree_err: result_data['error'] = f"Error parsing tree: {tree_err}"
                                else: result_data['error'] = "Solved status but no valid trees found."; result_data['synth_possible'] = False
                            else:
                                result_data['error'] = f"Status: {output_data.get('status', 'unknown')}"
                        elif not result_data['error']: # Если ошибки еще нет, но output_data получить не удалось
                               result_data['error'] = "Could not extract result dictionary from JSON"
                        # --- КОНЕЦ ИЗМЕНЕНИЙ ---
            except json.JSONDecodeError as json_err: result_data['error'] = f"JSON Decode Error: {json_err}"
            except Exception as parse_err: result_data['error'] = f"Error processing JSON: {parse_err}"
        else: result_data['error'] = "Output JSON file not found."

    except subprocess.TimeoutExpired: result_data['error'] = f"Timeout (> {AIZYNTH_TIMEOUT}s)"
    except IOError as io_err: result_data['error'] = f"File IO Error: {io_err}"
    except Exception as e: result_data['error'] = f"General error: {e}"; traceback.print_exc()
    finally:
        # Удаляем временные файлы
        if temp_smiles_filename and os.path.exists(temp_smiles_filename):
            try: os.remove(temp_smiles_filename)
            except OSError: pass
        if os.path.exists(temp_output_json):
            # Оставляем JSON файл если была ошибка (КРОМЕ "не решено") для отладки
            if result_data.get('error') and "Status: " not in str(result_data.get('error')):
                 print(f"--- Problematic JSON Output for SMILES: {smiles} (Error: {result_data['error']}) ---")
                 print(f"--- JSON file left for inspection: {temp_output_json} ---")
            else:
                 try: os.remove(temp_output_json)
                 except OSError: pass

    return result_data

# --- Остальной код скрипта (if __name__ == "__main__": и т.д.) без изменений ---
# ... (вставь его сюда из предыдущего ответа) ...

# --- 3. Основной блок выполнения ---
if __name__ == "__main__":

    # --- Парсинг аргументов ---
    parser = argparse.ArgumentParser(description='Запуск ретросинтетического анализа AiZynthFinder для SMILES из файла.')
    parser.add_argument('-i', '--input', type=str, required=True, help='Входной TXT или CSV файл со SMILES.')
    parser.add_argument('-o', '--output', type=str, default=DEFAULT_OUTPUT_FILENAME, help='Выходной CSV файл с результатами.')
    parser.add_argument('--smiles_col', type=str, default=None, help='Имя колонки со SMILES (если CSV).')
    parser.add_argument('--id_col', type=str, default=None, help='Имя колонки с ID (если есть в CSV).')

    args = parser.parse_args()
    input_filename = args.input
    output_filename = args.output
    smiles_column_name_arg = args.smiles_col
    index_column_name = args.id_col

    print("--- Старт Ретросинтетического Анализа с AiZynthFinder (CPU) ---")
    start_time_total = time.time()
    script_successful = True

    # Проверка конфига AiZynthFinder
    if not os.path.exists(AIZYNTH_CONFIG_PATH):
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Файл конфигурации AiZynthFinder не найден: {AIZYNTH_CONFIG_PATH}")
        exit(1)

    try:
        # --- Шаг 1: Загрузка данных ---
        # ... (Логика чтения файла остается такой же) ...
        print(f"\nШаг 1: Чтение файла '{input_filename}'...")
        try:
            smiles_col_to_use = DEFAULT_SMILES_COL
            df_input = pd.read_csv(input_filename, header=None, names=[smiles_col_to_use], skip_blank_lines=True)
            print(f"Файл прочитан как TXT без заголовка. Используется колонка '{smiles_col_to_use}'.")
        except Exception as read_e:
            # print(f"Ошибка при чтении как TXT: {read_e}. Пробуем как CSV...")
            try:
                df_input = pd.read_csv(input_filename)
                if smiles_column_name_arg and smiles_column_name_arg in df_input.columns: smiles_col_to_use = smiles_column_name_arg
                else:
                    potential_cols = [c for c in df_input.columns if 'smiles' in c.lower()]
                    if potential_cols: smiles_col_to_use = potential_cols[0]
                    elif len(df_input.columns) > 0: smiles_col_to_use = df_input.columns[0]
                    else: raise ValueError("Не найдена колонка SMILES")
                print(f"Файл прочитан как CSV. Используется колонка '{smiles_col_to_use}'.")
            except Exception as read_e2: raise Exception(f"Не удалось прочитать файл '{input_filename}': {read_e2}")

        print(f"Загружено {len(df_input)} строк.")
        df_input.dropna(subset=[smiles_col_to_use], inplace=True)
        df_input = df_input[df_input[smiles_col_to_use].astype(str).str.strip() != '']
        df_input = df_input.drop_duplicates(subset=[smiles_col_to_use], keep='first')
        df_input.reset_index(drop=True, inplace=True)
        print(f"Осталось {len(df_input)} уникальных непустых SMILES для анализа.")
        if df_input.empty: raise ValueError("Нет валидных SMILES для анализа.")

        use_df_index = False
        index_col_to_use = args.id_col
        if index_col_to_use is None or index_col_to_use not in df_input.columns:
             df_input['temp_index_for_merge'] = range(len(df_input))
             index_col_to_use = 'temp_index_for_merge'
             use_df_index = True

        # --- Шаг 2: Запуск анализа ---
        results_retrosynthesis = []
        print(f"\nШаг 2: Запуск анализа для {len(df_input)} молекул...")
        for index, row in tqdm(df_input.iterrows(), total=len(df_input), desc="Ретросинтез (AiZynth)"):
            smiles_to_analyze = row[smiles_col_to_use]
            # Пропускаем слишком длинные SMILES
            if len(smiles_to_analyze) > 500:
                 print(f"  Пропуск слишком длинного SMILES (>500): {smiles_to_analyze[:30]}...")
                 retro_result = {'synth_possible': False, 'synth_steps': np.nan, 'synth_score': np.nan, 'error': 'SMILES too long'}
            else:
                retro_result = run_single_retrosynthesis_aizynth(smiles_to_analyze)

            retro_result[index_col_to_use] = row[index_col_to_use]
            results_retrosynthesis.append(retro_result)
            # Выводим результат для каждой молекулы (исправлено форматирование score)
            score_str = f"{retro_result['synth_score']:.3f}" if isinstance(retro_result.get('synth_score'), (int, float)) and not pd.isna(retro_result.get('synth_score')) else 'N/A'
            error_str = retro_result.get('error', 'None') or 'None'
            print(f"  {index + 1}/{len(df_input)}: {smiles_to_analyze[:30]}... -> Possible={retro_result['synth_possible']} (Steps={retro_result['synth_steps']}) Score={score_str} Error: {error_str}")

        print("\nРетросинтетический анализ завершен.")
        df_retro_results = pd.DataFrame(results_retrosynthesis)

        # --- Шаг 3: Объединение результатов ---
        print("\nШаг 3: Объединение результатов...")
        df_final_with_retro = pd.merge(df_input, df_retro_results, on=index_col_to_use, how='left')
        if use_df_index: df_final_with_retro.drop(columns=['temp_index_for_merge'], inplace=True)
        print(f"Результаты объединены. Итоговый размер: {df_final_with_retro.shape}")

        # Статистика
        if 'synth_possible' in df_final_with_retro.columns:
             possible_count = df_final_with_retro['synth_possible'].sum()
             print(f"  - Найдено путей синтеза для: {possible_count} из {len(df_final_with_retro)} ({possible_count / len(df_final_with_retro) * 100:.1f}%)")
             if possible_count > 0 and 'synth_steps' in df_final_with_retro.columns:
                  valid_steps = pd.to_numeric(df_final_with_retro.loc[df_final_with_retro['synth_possible'] == True, 'synth_steps'], errors='coerce').dropna()
                  if not valid_steps.empty:
                       avg_steps = valid_steps.mean()
                       print(f"  - Среднее количество стадий (для успешных): {avg_steps:.1f}")

        # --- Шаг 4: Сохранение ---
        print("\nШаг 4: Сохранение итогового файла...")
        try:
            # Сохраняем исходный SMILES и результаты ретросинтеза + другие колонки из исходного файла
            cols_to_save_retro = [col for col in df_retro_results.columns if col != index_col_to_use]
            df_final_with_retro[[smiles_col_to_use] + [c for c in df_input.columns if c not in [smiles_col_to_use, 'temp_index_for_merge']] + cols_to_save_retro].to_csv(output_filename, index=False)
            print(f"  - Файл с результатами ретросинтеза сохранен: {output_filename}")
        except Exception as e: print(f"  - Ошибка сохранения файла: {e}")

    # --- Обработка ошибок ---
    except FileNotFoundError as fnf_e: print(f"\nКРИТИЧЕСКАЯ ОШИБКА: Файл не найден - {fnf_e}"); script_successful = False
    except ValueError as ve: print(f"\nКРИТИЧЕСКАЯ ОШИБКА ДАННЫХ: {ve}"); script_successful = False
    except Exception as e: print(f"\nПроизошла непредвиденная КРИТИЧЕСКАЯ ошибка: {e}"); traceback.print_exc(); script_successful = False
    finally:
        end_time_total = time.time()
        print(f"\n--- Завершение Ретросинтетического Анализа ---")
        if script_successful: print("Скрипт завершился успешно.")
        else: print("Скрипт завершился с ошибками.")
        print(f"Общее время выполнения: {(end_time_total - start_time_total):.2f} секунд.")
        warnings.filterwarnings("default")
        # rdBase.EnableLog('rdApp.error')