import pandas as pd
from tqdm.auto import tqdm
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gc # Garbage Collector для управления памятью
import warnings

# --- RDKit Imports ---
RDKIT_AVAILABLE = False
BUTINA_AVAILABLE = False # Уточним позже
try:
    from rdkit import Chem
    from rdkit.Chem.rdchem import Mol
    from rdkit.Chem import Draw, Descriptors, AllChem, QED, DataStructs

    # Более надежный способ импорта Butina
    try:
        from rdkit.ML.Cluster import Butina
        BUTINA_AVAILABLE = True
        print("INFO: Успешно импортирован Butina из rdkit.ML.Cluster.")
    except ImportError:
         # Пробуем старый путь, если новый не сработал (для старых версий RDKit)
         try:
             from rdkit.Chem.Fingerprints import ClusterData # Не совсем Butina, но кластеризация
             # Этот путь менее стандартный для Butina, лучше обновить RDKit
             print("ПРЕДУПРЕЖДЕНИЕ: Используется ClusterData из Fingerprints. Рекомендуется обновить RDKit для использования rdkit.ML.Cluster.Butina.")
             # Не устанавливаем BUTINA_AVAILABLE=True, т.к. это может быть не совсем то
         except ImportError:
             print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать модуль кластеризации Butina. Кластеризация будет недоступна.")

    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import RDConfig
    from rdkit import rdBase
    rdBase.DisableLog('rdApp.warning') # Отключаем подробные логи RDKit
    print(f"INFO: RDKit version: {rdBase.rdkitVersion}")
    RDKIT_AVAILABLE = True
except ImportError:
    print("ОШИБКА: RDKit не найден. Пожалуйста, установите RDKit. Скрипт не может продолжить.")
    exit() # Выход, если RDKit недоступен

# --- SA Score Import ---
SA_SCORER_AVAILABLE = False
if RDKIT_AVAILABLE:
    try:
        sascorer_path = os.path.join(RDConfig.RDContribDir, 'SA_Score')
        if sascorer_path not in sys.path:
             sys.path.append(sascorer_path)
        import sascorer
        SA_SCORER_AVAILABLE = True
        print("INFO: Модуль SA_Score успешно импортирован.")
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: Модуль SA_Score не найден. SA Score не будет рассчитан.")
    except Exception as e:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка при импорте SA_Score: {e}. SA Score не будет рассчитан.")

# --- Конфигурация и Константы ---
# Используем пути, предоставленные пользователем
# Лучше определить базовую директорию или использовать абсолютные пути
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else '.' # Определяем директорию скрипта или текущую
DATA_DIR = os.path.join(BASE_DIR, 'data', 'chembl') # Создаем путь к директории данных

INITIAL_DATA_FILE = '/content/Neftekod_initial_data.csv' # Оставляем как есть, если это точно верный путь
CHEMBL_DATA_FILE = '/content/chembl_35_chemreps.txt' # Оставляем как есть
CHEMBL_FILTERED_OUTPUT_FILE = os.path.join(DATA_DIR, 'chembl777_filtered_for_generator.csv')

# Убедимся, что директория для выходного файла существует
os.makedirs(DATA_DIR, exist_ok=True)
print(f"INFO: Директория для выходных данных: {DATA_DIR}")

# Параметры фильтрации и анализа
MAX_MW = 1000.0
MIN_LOGP = 1.0
ALLOWED_ATOMS = {'C', 'H', 'O', 'N', 'P', 'S'}
CLUSTER_TANIMOTO_SIMILARITY_CUTOFF = 0.7 # Порог СХОЖЕСТИ
FP_RADIUS = 2
FP_NBITS = 2048
CHEMBL_CHUNK_SIZE = 50000 # Уменьшил для ускорения тестов, можно вернуть 100000

# --- Вспомогательные Функции (Улучшенные и унифицированные) ---

def safe_mol_from_smiles(smiles: str) -> Mol | None:
    """Безопасно создает Mol объект из SMILES, возвращает None при ошибке."""
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        # Дополнительная проверка, что RDKit не вернул None молча
        if mol is None:
             return None
        # Опционально: Раскомментируйте для более строгой проверки (может отсеять некоторые валидные, но "странные" структуры)
        # Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None

# Pre-compile SMARTS patterns for efficiency
try:
    PHENOL_SMARTS = Chem.MolFromSmarts('[#8&X2H1]-[c]') # Фенольный OH
    AROMATIC_AMINE_SMARTS = Chem.MolFromSmarts('[#7;!$([#7]=O);!$([#7]~[O,N])]-[a]') # Ароматический амин (не амид/нитрозо)
    if PHENOL_SMARTS is None or AROMATIC_AMINE_SMARTS is None:
        raise ValueError("Не удалось скомпилировать SMARTS паттерны.")
    print("INFO: SMARTS паттерны скомпилированы.")
except Exception as e:
     print(f"ОШИБКА: Не удалось скомпилировать SMARTS: {e}. Идентификация целевого класса не будет работать.")
     PHENOL_SMARTS = None
     AROMATIC_AMINE_SMARTS = None

def check_smarts_match(mol: Mol | None, pattern: Mol | None) -> bool:
    """Проверяет наличие SMARTS паттерна в молекуле."""
    if mol is None or pattern is None: return False
    try:
        return mol.HasSubstructMatch(pattern)
    except Exception: return False # На случай редких ошибок RDKit

def is_phenol(mol: Mol | None) -> bool:
    """Проверяет, является ли молекула фенолом."""
    return check_smarts_match(mol, PHENOL_SMARTS)

def is_aromatic_amine(mol: Mol | None) -> bool:
    """Проверяет, является ли молекула ароматическим амином."""
    return check_smarts_match(mol, AROMATIC_AMINE_SMARTS)

def is_target_class(mol: Mol | None) -> bool:
    """Проверяет, является ли молекула фенолом ИЛИ ароматическим амином."""
    if mol is None: return False
    # Проверяем только если SMARTS были успешно скомпилированы
    has_phenol = is_phenol(mol) if PHENOL_SMARTS else False
    has_amine = is_aromatic_amine(mol) if AROMATIC_AMINE_SMARTS else False
    return has_phenol or has_amine

def has_radical_electrons(mol: Mol | None) -> bool:
    """Проверяет наличие радикальных электронов."""
    if mol is None: return True # Считаем None проблемой
    try:
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0: return True
        return False
    except Exception: return True # Ошибка -> считаем проблемой

def is_neutral_and_stable(mol: Mol | None) -> bool:
    """Проверяет нейтральность заряда и отсутствие радикалов."""
    if mol is None: return False
    try:
        # Проверяем и формальный заряд, и радикалы
        return Chem.GetFormalCharge(mol) == 0 and not has_radical_electrons(mol)
    except Exception: return False # Ошибка -> не пройдено

def meets_mw_criteria(mol: Mol | None) -> bool:
    """Проверяет молекулярный вес."""
    if mol is None: return False
    try:
        mw = Descriptors.MolWt(mol)
        # Проверяем, что mw - число, не NaN, > 0 и в пределах лимита
        return isinstance(mw, (int, float)) and not np.isnan(mw) and 0 < mw <= MAX_MW
    except Exception: return False

def contains_only_allowed_atoms(mol: Mol | None) -> bool:
    """Проверяет состав атомов."""
    if mol is None: return False
    try:
        return all(atom.GetSymbol() in ALLOWED_ATOMS for atom in mol.GetAtoms())
    except Exception: return False

def meets_logp_criteria(mol: Mol | None) -> bool:
    """Проверяет LogP."""
    if mol is None: return False
    try:
        logp = Descriptors.MolLogP(mol)
        # Проверяем, что logp - число, не NaN и больше минимума
        return isinstance(logp, (int, float)) and not np.isnan(logp) and logp > MIN_LOGP
    except Exception: # Включая ValueError от RDKit
        return False

def calculate_sa_score(mol: Mol | None) -> float | None:
    """Расчет SA Score с обработкой ошибок."""
    if mol is None or not SA_SCORER_AVAILABLE: return np.nan # Возвращаем NaN для консистентности
    try:
        sa = sascorer.calculateScore(mol)
        return sa if np.isfinite(sa) else np.nan # Проверяем на NaN/inf
    except Exception: return np.nan

def calculate_qed_score(mol: Mol | None) -> float | None:
    """Расчет QED с обработкой ошибок."""
    if mol is None: return np.nan
    try:
        qed_val = QED.qed(mol)
        return qed_val if np.isfinite(qed_val) else np.nan
    except Exception: return np.nan

def calculate_gasteiger_charges(mol: Mol | None) -> pd.Series:
    """Расчет зарядов Gasteiger с улучшенной обработкой ошибок."""
    charges = {'O_phenol_avg': np.nan, 'H_phenol_avg': np.nan,
               'N_amine_avg': np.nan, 'H_amine_avg': np.nan}
    if mol is None: return pd.Series(charges)

    try:
        # Пытаемся рассчитать заряды, не падаем при ошибке
        status = AllChem.ComputeGasteigerCharges(mol, nIter=15, throwOnFailure=False)

        # Проверяем, появились ли свойства с зарядами
        has_charges = any(atom.HasProp('_GasteigerCharge') for atom in mol.GetAtoms())
        if not has_charges:
            # print(f"DEBUG: Не удалось рассчитать Gasteiger Charges для {Chem.MolToSmiles(mol)}")
            return pd.Series(charges) # Возвращаем NaN, если расчет не удался

        o_phenol_charges, h_phenol_charges = [], []
        # Проверяем класс только если SMARTS доступны
        if PHENOL_SMARTS and is_phenol(mol):
            phenol_matches = mol.GetSubstructMatches(PHENOL_SMARTS)
            for match in phenol_matches:
                 o_idx = match[0]
                 atom_o = mol.GetAtomWithIdx(o_idx)
                 if atom_o.HasProp('_GasteigerCharge'):
                     try:
                         o_charge = float(atom_o.GetProp('_GasteigerCharge'))
                         if np.isfinite(o_charge): o_phenol_charges.append(o_charge)
                     except (ValueError, TypeError): pass # Игнорируем, если заряд не float

                     for neighbor in atom_o.GetNeighbors():
                         if neighbor.GetSymbol() == 'H' and neighbor.HasProp('_GasteigerCharge'):
                             try:
                                 h_charge = float(neighbor.GetProp('_GasteigerCharge'))
                                 if np.isfinite(h_charge): h_phenol_charges.append(h_charge)
                             except (ValueError, TypeError): pass
                             break # Один H у фенола

        n_amine_charges, h_amine_charges_means = [], []
        # Проверяем класс только если SMARTS доступны
        if AROMATIC_AMINE_SMARTS and is_aromatic_amine(mol):
            amine_matches = mol.GetSubstructMatches(AROMATIC_AMINE_SMARTS)
            for match in amine_matches:
                n_idx = match[0]
                atom_n = mol.GetAtomWithIdx(n_idx)
                if atom_n.HasProp('_GasteigerCharge'):
                    try:
                        n_charge = float(atom_n.GetProp('_GasteigerCharge'))
                        if np.isfinite(n_charge): n_amine_charges.append(n_charge)
                    except (ValueError, TypeError): pass

                    temp_h_charges = []
                    for neighbor in atom_n.GetNeighbors():
                        if neighbor.GetSymbol() == 'H' and neighbor.HasProp('_GasteigerCharge'):
                             try:
                                 h_charge = float(neighbor.GetProp('_GasteigerCharge'))
                                 if np.isfinite(h_charge): temp_h_charges.append(h_charge)
                             except (ValueError, TypeError): pass
                    if temp_h_charges:
                         # Используем nanmean на случай ошибок в отдельных H зарядах
                         with warnings.catch_warnings(): # Подавляем RuntimWarning для пустого среза
                             warnings.simplefilter("ignore", category=RuntimeWarning)
                             mean_h_charge = np.nanmean(temp_h_charges)
                         if np.isfinite(mean_h_charge):
                             h_amine_charges_means.append(mean_h_charge)

        # Усредняем, только если есть данные
        if o_phenol_charges: charges['O_phenol_avg'] = np.mean(o_phenol_charges)
        if h_phenol_charges: charges['H_phenol_avg'] = np.mean(h_phenol_charges)
        if n_amine_charges: charges['N_amine_avg'] = np.mean(n_amine_charges)
        if h_amine_charges_means: charges['H_amine_avg'] = np.mean(h_amine_charges_means)

    except Exception as e:
        # Тихо пропускаем ошибку расчета зарядов для данной молекулы
        # Можно добавить логирование при необходимости
        # print(f"ПРЕДУПРЕЖДЕНИЕ: Ошибка расчета зарядов для SMILES '{Chem.MolToSmiles(mol) if mol else 'Invalid'}'. Ошибка: {e}")
        pass
    return pd.Series(charges)

# --- Инициализация TQDM для Pandas ---
tqdm.pandas(desc="Pandas Apply")

# === ЭТАП 1: Анализ Исходных Данных (df) ===
print("\n" + "="*50 + "\n=== ЭТАП 1: Анализ Исходных Данных (df) ===\n" + "="*50)
df_processed = pd.DataFrame() # Инициализируем пустой DataFrame для результата
stage1_success = False
try:
    if not os.path.exists(INITIAL_DATA_FILE):
        raise FileNotFoundError(f"Файл {INITIAL_DATA_FILE} не найден.")

    df_initial = pd.read_csv(INITIAL_DATA_FILE)
    print(f"INFO: Загружено {len(df_initial)} записей из {INITIAL_DATA_FILE}")

    if 'Smiles' not in df_initial.columns:
        raise ValueError("Колонка 'Smiles' отсутствует в исходном файле.")

    # 1. Создание Mol объектов
    print("INFO: [1.1] Создание RDKit Mol объектов...")
    df_initial['Mol'] = df_initial['Smiles'].progress_apply(safe_mol_from_smiles)

    n_initial = len(df_initial)
    n_invalid = df_initial['Mol'].isnull().sum()
    n_valid = n_initial - n_invalid

    if n_invalid > 0:
        print(f"ПРЕДУПРЕЖДЕНИЕ: Найдено {n_invalid} невалидных/проблемных SMILES из {n_initial}.")
        print("  -> Примеры невалидных:", df_initial.loc[df_initial['Mol'].isnull(), 'Smiles'].head(5).tolist())

    df_initial.dropna(subset=['Mol'], inplace=True)
    df_initial.reset_index(drop=True, inplace=True)
    print(f"INFO: Используется {n_valid} валидных Mol объектов.")

    if n_valid == 0:
        print("ОШИБКА: Нет валидных молекул в исходном файле. Пропуск Этапа 1.")

    else:
        # 2. Применение Фильтров
        print("\nINFO: [1.2] Применение фильтров...")

        # Определяем фильтры
        filters = {
            "target_class": is_target_class,
            "neutral_stable": is_neutral_and_stable,
            "allowed_atoms": contains_only_allowed_atoms,
            "mw_criteria": meets_mw_criteria,
            "logp_criteria": meets_logp_criteria,
        }

        passed_mask = pd.Series(True, index=df_initial.index)
        filter_results = {}

        for name, func in filters.items():
            print(f"INFO: Применение фильтра '{name}'...")
            current_mask = df_initial['Mol'].progress_apply(func)
            filter_results[name] = current_mask.sum()
            passed_mask &= current_mask

            # Опционально: показать, сколько не прошло
            if not current_mask.all():
                 failed_count = (~current_mask).sum()
                 # print(f"  -> {failed_count} молекул не прошло '{name}'")
                 # print(f"  -> Примеры не прошедших '{name}':", df_initial.loc[~current_mask, 'Smiles'].head(3).tolist())


        print("\n--- Результаты фильтрации для df ---")
        print(f"Исходное количество валидных молекул: {n_valid}")
        for name, count in filter_results.items():
            pass_perc = (count / n_valid * 100) if n_valid > 0 else 0
            print(f"Прошло '{name}': {count} / {n_valid} ({pass_perc:.1f}%)")

        df_processed = df_initial[passed_mask].copy()
        n_processed = len(df_processed)
        n_filtered_out = n_valid - n_processed

        print(f"\nINFO: Фильтрация завершена. Осталось молекул: {n_processed}")
        if n_filtered_out > 0:
            print(f"INFO: Отфильтровано: {n_filtered_out} молекул.")
            # print("  -> Примеры отфильтрованных:", df_initial.loc[~passed_mask, 'Smiles'].head(5).tolist())
        df_processed.reset_index(drop=True, inplace=True)

        if not df_processed.empty:
            # 3. Расчет Дескрипторов и Зарядов
            print("\nINFO: [1.3] Расчет дескрипторов и зарядов...")
            df_processed['mol_weight'] = df_processed['Mol'].progress_apply(lambda m: Descriptors.MolWt(m) if m else np.nan)
            df_processed['log_p'] = df_processed['Mol'].progress_apply(lambda m: Descriptors.MolLogP(m) if m else np.nan)
            df_processed['SA_Score'] = df_processed['Mol'].progress_apply(calculate_sa_score)
            df_processed['QED'] = df_processed['Mol'].progress_apply(calculate_qed_score)

            print("INFO: Расчет зарядов Gasteiger...")
            charge_df = df_processed['Mol'].progress_apply(calculate_gasteiger_charges)
            # Присоединяем результаты расчета зарядов
            df_processed = pd.concat([df_processed, charge_df], axis=1)
            print("INFO: Дескрипторы и заряды рассчитаны.")

            # 4. Классификация и Кластеризация
            print("\nINFO: [1.4] Классификация и Кластеризация...")
            if PHENOL_SMARTS: df_processed['is_phenol'] = df_processed['Mol'].progress_apply(is_phenol)
            else: df_processed['is_phenol'] = False
            if AROMATIC_AMINE_SMARTS: df_processed['is_amine'] = df_processed['Mol'].progress_apply(is_aromatic_amine)
            else: df_processed['is_amine'] = False
            print(f"INFO: Классификация -> Фенолы: {df_processed['is_phenol'].sum()}, Амины: {df_processed['is_amine'].sum()}")

            # Кластеризация Butina
            df_processed['ClusterID'] = -1 # ID кластера по умолчанию
            if BUTINA_AVAILABLE and n_processed > 1:
                print("INFO: Генерация фингерпринтов для кластеризации...")
                fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)
                fps = [fpgen.GetFingerprint(m) for m in tqdm(df_processed['Mol'], desc="Fingerprinting")]

                # Удаляем None фингерпринты, если есть (хотя после dropna('Mol') их быть не должно)
                valid_fps_indices = [i for i, fp in enumerate(fps) if fp is not None]
                valid_fps = [fps[i] for i in valid_fps_indices]
                n_valid_fps = len(valid_fps)

                if n_valid_fps > 1:
                    print(f"INFO: Расчет матрицы расстояний для {n_valid_fps} фингерпринтов...")
                    dists = []
                    for i in tqdm(range(1, n_valid_fps), desc="Calculating Distances"):
                        sims = DataStructs.BulkTanimotoSimilarity(valid_fps[i], valid_fps[:i])
                        dists.extend([1 - x for x in sims]) # Tanimoto Distance

                    print("INFO: Запуск кластеризации Butina...")
                    dist_cutoff = 1.0 - CLUSTER_TANIMOTO_SIMILARITY_CUTOFF
                    clusters = Butina.ClusterData(dists, n_valid_fps, dist_cutoff, isDistData=True)

                    print(f"INFO: Найдено {len(clusters)} кластеров.")
                    cluster_sizes = {}
                    # Присваиваем ID кластеров обратно в DataFrame
                    for cluster_id, member_indices in enumerate(clusters):
                        original_indices = [valid_fps_indices[i] for i in member_indices]
                        cluster_sizes[cluster_id] = len(original_indices)
                        # Используем .iloc для присвоения по числовому индексу
                        df_processed.iloc[original_indices, df_processed.columns.get_loc('ClusterID')] = cluster_id

                    print("INFO: Топ 10 размеров кластеров:", sorted(cluster_sizes.items(), key=lambda item: item[1], reverse=True)[:10])
                else:
                    print("ПРЕДУПРЕЖДЕНИЕ: Недостаточно валидных фингерпринтов для кластеризации.")
            elif not BUTINA_AVAILABLE:
                print("ПРЕДУПРЕЖДЕНИЕ: Кластеризация пропущена - модуль Butina недоступен.")
            else: # n_processed <= 1
                 print("INFO: Кластеризация пропущена - менее 2 молекул.")

            # 5. Статистика и Визуализация
            print("\nINFO: [1.5] Финальная статистика по дескрипторам:")
            cols_to_describe = ['mol_weight', 'log_p', 'SA_Score', 'QED',
                                'O_phenol_avg', 'H_phenol_avg', 'N_amine_avg', 'H_amine_avg']
            existing_cols = [col for col in cols_to_describe if col in df_processed.columns and df_processed[col].notna().any()]
            if existing_cols:
                 print(df_processed[existing_cols].describe().round(3))
            else:
                 print("Нет данных для статистики.")

            print("\nINFO: [1.6] Генерация визуализаций...")
            # Гистограммы
            try:
                plot_cols = ['mol_weight', 'log_p', 'SA_Score', 'QED', 'O_phenol_avg', 'N_amine_avg']
                valid_plot_cols = [col for col in plot_cols if col in existing_cols] # Берем только существующие и не пустые

                if valid_plot_cols:
                    n_plots = len(valid_plot_cols)
                    n_cols_fig = min(3, n_plots)
                    n_rows_fig = (n_plots + n_cols_fig - 1) // n_cols_fig
                    fig, axes = plt.subplots(n_rows_fig, n_cols_fig, figsize=(5 * n_cols_fig, 4 * n_rows_fig), squeeze=False)
                    axes = axes.ravel()

                    for i, col in enumerate(valid_plot_cols):
                        data_to_plot = df_processed[col].dropna()
                        if len(data_to_plot) > 0:
                             sns.histplot(data_to_plot, ax=axes[i], kde=False, bins=max(5, min(len(data_to_plot), 15)))
                             axes[i].set_title(f'Распределение {col}')
                        else:
                             axes[i].set_title(f'Распределение {col} (нет данных)')
                             axes[i].text(0.5, 0.5, 'Нет данных', ha='center', va='center')

                    for j in range(n_plots, len(axes)): axes[j].set_visible(False) # Скрываем лишние

                    plt.tight_layout()
                    hist_filename = os.path.join(DATA_DIR, 'df_initial_descriptors_histograms.png')
                    plt.savefig(hist_filename, dpi=150)
                    print(f"INFO: Гистограммы сохранены: {hist_filename}")
                    # plt.show() # Раскомментируйте, если нужен показ в интерактивном режиме
                    plt.close(fig)
                else:
                    print("INFO: Нет данных для построения гистограмм.")
            except Exception as e:
                print(f"ОШИБКА: Не удалось сгенерировать гистограммы: {e}")

            # Сетка молекул
            print("INFO: Генерация сетки примеров молекул...")
            try:
                num_to_draw = min(9, n_processed)
                if num_to_draw > 0:
                     sample_df = df_processed.head(num_to_draw)
                     mols_to_draw = sample_df['Mol'].tolist()
                     legends = []
                     for _, row in sample_df.iterrows():
                         legend = f"{row['Smiles'][:35]}" + ('...' if len(row['Smiles']) > 35 else '')
                         if 'ClusterID' in row and row['ClusterID'] != -1:
                             legend += f"\nКластер: {row['ClusterID']}"
                         legends.append(legend)

                     img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=3, subImgSize=(300, 300), legends=legends, useSVG=False)
                     if img:
                         grid_filename = os.path.join(DATA_DIR, 'df_initial_sample_molecules.png')
                         img.save(grid_filename)
                         print(f"INFO: Сетка молекул сохранена: {grid_filename}")
                     else:
                         print("ПРЕДУПРЕЖДЕНИЕ: Draw.MolsToGridImage вернул None.")
                else:
                     print("INFO: Нет молекул для отрисовки сетки.")
            except Exception as e:
                 print(f"ОШИБКА: Не удалось сгенерировать сетку молекул: {e}")

        stage1_success = True # Помечаем успешное завершение этапа
        print("\n--- Этап 1 Завершен ---")

except FileNotFoundError as fnf_error:
    print(f"ОШИБКА [Этап 1]: {fnf_error}")
except ValueError as ve:
    print(f"ОШИБКА Данных [Этап 1]: {ve}")
except Exception as e:
    print(f"Непредвиденная ОШИБКА [Этап 1]: {e}")
    import traceback
    traceback.print_exc()

# Очистка памяти после Этапа 1
if 'df_initial' in locals(): del df_initial
gc.collect()

# === ЭТАП 2: Фильтрация ChEMBL (Исправленная Логика) ===
print("\n" + "="*50 + "\n=== ЭТАП 2: Фильтрация ChEMBL ===\n" + "="*50)
run_chembl_processing = False
if not os.path.exists(CHEMBL_DATA_FILE):
    print(f"ОШИБКА: Файл ChEMBL не найден: {CHEMBL_DATA_FILE}. Пропуск Этапа 2.")
elif not RDKIT_AVAILABLE:
     print(f"ОШИБКА: RDKit недоступен. Пропуск Этапа 2.")
else:
    # Проверка и запрос на перезапись выходного файла
    if os.path.exists(CHEMBL_FILTERED_OUTPUT_FILE):
        try:
            overwrite = input(f"ПРЕДУПРЕЖДЕНИЕ: Файл {CHEMBL_FILTERED_OUTPUT_FILE} существует. Перезаписать? (y/n): ").strip().lower()
            if overwrite == 'y':
                print("INFO: Файл будет перезаписан.")
                try:
                    os.remove(CHEMBL_FILTERED_OUTPUT_FILE)
                    run_chembl_processing = True
                except OSError as e:
                    print(f"ОШИБКА: Не удалось удалить существующий файл: {e}. Пропуск Этапа 2.")
            else:
                print("INFO: Фильтрация ChEMBL пропущена (файл не перезаписан).")
        except EOFError: # Обработка для неинтерактивных сред
             print("ПРЕДУПРЕЖДЕНИЕ: Неинтерактивная среда. Перезапись существующего файла.")
             try:
                 os.remove(CHEMBL_FILTERED_OUTPUT_FILE)
                 run_chembl_processing = True
             except OSError as e:
                 print(f"ОШИБКА: Не удалось удалить существующий файл: {e}. Пропуск Этапа 2.")
    else:
        run_chembl_processing = True # Файла нет, создаем

if run_chembl_processing:
    print(f"INFO: Старт фильтрации ChEMBL из {CHEMBL_DATA_FILE}")
    print(f"INFO: Размер чанка: {CHEMBL_CHUNK_SIZE} строк.")
    print(f"INFO: Выходной файл: {CHEMBL_FILTERED_OUTPUT_FILE}")

    total_processed_lines = 0
    total_passed_molecules = 0
    first_chunk_written = False
    start_time_chembl = pd.Timestamp.now()

    try:
        # Итератор по чанкам
        reader = pd.read_csv(CHEMBL_DATA_FILE, sep='\t', chunksize=CHEMBL_CHUNK_SIZE,
                             iterator=True, low_memory=False, on_bad_lines='warn') # Предупреждать о плохих строках

        # Оценка количества чанков для tqdm (опционально)
        try:
            print("INFO: Оценка общего количества строк (может занять время)...")
            # Открываем файл безопасно с указанием кодировки и обработкой ошибок
            with open(CHEMBL_DATA_FILE, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines_approx = sum(1 for _ in f)
            n_chunks_approx = (total_lines_approx + CHEMBL_CHUNK_SIZE - 1) // CHEMBL_CHUNK_SIZE
            print(f"INFO: Примерно {total_lines_approx} строк в {n_chunks_approx} чанках.")
        except Exception as e:
            print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось оценить количество строк: {e}. TQDM будет без общего числа.")
            n_chunks_approx = None # Не удалось оценить

        # Обработка чанков с tqdm
        with tqdm(reader, total=n_chunks_approx, desc="Обработка ChEMBL") as pbar:
            for i, chunk_df in enumerate(pbar):
                chunk_start_time = pd.Timestamp.now()
                initial_chunk_size = len(chunk_df)
                total_processed_lines += initial_chunk_size

                # 1. Поиск колонки SMILES
                smiles_col = None
                for col in ['canonical_smiles', 'smiles', 'SMILES']:
                     if col in chunk_df.columns:
                         smiles_col = col
                         break
                if not smiles_col:
                    pbar.set_postfix_str("Пропущен (нет SMILES)")
                    continue # Пропускаем чанк без SMILES

                # 2. Создание Mol объектов (более эффективно)
                # Используем safe_mol_from_smiles и dropna
                chunk_df['Mol'] = chunk_df[smiles_col].apply(safe_mol_from_smiles) # Не progress_apply внутри цикла
                chunk_df.dropna(subset=['Mol'], inplace=True)

                n_valid_in_chunk = len(chunk_df)
                if n_valid_in_chunk == 0:
                    pbar.set_postfix_str("Пропущен (нет валидных Mol)")
                    del chunk_df; gc.collect(); continue # Если нет валидных Mol, идем дальше

                # 3. Применение ВСЕХ фильтров (ВКЛЮЧАЯ is_target_class)
                # Создаем маску поэтапно для читаемости (оптимизатор Python справится)
                mask = chunk_df['Mol'].apply(is_target_class) # <-- ДОБАВЛЕН ФИЛЬТР КЛАССА
                mask &= chunk_df['Mol'].apply(is_neutral_and_stable)
                mask &= chunk_df['Mol'].apply(meets_mw_criteria)
                mask &= chunk_df['Mol'].apply(contains_only_allowed_atoms)
                mask &= chunk_df['Mol'].apply(meets_logp_criteria)

                # Выбираем только нужную колонку из прошедших фильтр строк
                filtered_chunk = chunk_df.loc[mask, [smiles_col]].copy()
                # Переименовываем для единообразия
                filtered_chunk.rename(columns={smiles_col: 'canonical_smiles'}, inplace=True)

                passed_in_chunk = len(filtered_chunk)
                total_passed_molecules += passed_in_chunk

                # 4. Запись в файл
                if passed_in_chunk > 0:
                    if not first_chunk_written:
                        filtered_chunk.to_csv(CHEMBL_FILTERED_OUTPUT_FILE, index=False, mode='w', header=True)
                        first_chunk_written = True
                    else:
                        filtered_chunk.to_csv(CHEMBL_FILTERED_OUTPUT_FILE, index=False, mode='a', header=False)

                # 5. Обновление статистики и очистка
                chunk_end_time = pd.Timestamp.now()
                chunk_duration = (chunk_end_time - chunk_start_time).total_seconds()
                if total_processed_lines > 0:
                    pass_rate = (total_passed_molecules / total_processed_lines) * 100
                    pbar.set_postfix({
                        "Passed": f"{total_passed_molecules}",
                        "Rate": f"{pass_rate:.3f}%",
                        "ChunkTime": f"{chunk_duration:.1f}s"
                        }, refresh=True) # refresh=True может быть полезно

                del chunk_df, filtered_chunk, mask
                gc.collect() # Принудительная сборка мусора после чанка

        end_time_chembl = pd.Timestamp.now()
        total_duration_chembl = (end_time_chembl - start_time_chembl).total_seconds()
        print("\n--- Результаты Фильтрации ChEMBL ---")
        print(f"Обработка завершена за: {total_duration_chembl:.2f} сек")
        print(f"Всего обработано строк (приблизительно): {total_processed_lines}")
        print(f"Прошло все фильтры (включая класс): {total_passed_molecules}")
        if total_processed_lines > 0:
            final_pass_rate = (total_passed_molecules / total_processed_lines) * 100
            print(f"Итоговый процент прошедших: {final_pass_rate:.3f}%")

        if first_chunk_written:
             print(f"Результат сохранен в: {CHEMBL_FILTERED_OUTPUT_FILE}")
             # Проверка первых строк файла
             try:
                 print("\n--- Первые 5 строк отфильтрованного файла ChEMBL ---")
                 df_chembl_head = pd.read_csv(CHEMBL_FILTERED_OUTPUT_FILE, nrows=5)
                 print(df_chembl_head)
                 print("-" * 50)
             except Exception as e:
                 print(f"ПРЕДУПРЕЖДЕНИЕ: Не удалось прочитать шапку выходного файла: {e}")
        else:
            print("INFO: Ни одна молекула не прошла фильтры. Выходной файл не создан.")

    except pd.errors.ParserError as pe:
         print(f"ОШИБКА [Этап 2]: Ошибка парсинга файла ChEMBL. Проверьте разделитель и формат. {pe}")
    except FileNotFoundError: # Повторная проверка на всякий случай
         print(f"ОШИБКА [Этап 2]: Файл {CHEMBL_DATA_FILE} не найден во время обработки.")
    except Exception as e:
         print(f"Непредвиденная ОШИБКА [Этап 2] при обработке ChEMBL: {e}")
         import traceback
         traceback.print_exc()


# === Следующие Шаги Плана === (Остаются актуальными)
print("\n" + "="*50 + "\n=== Следующие Шаги ===\n" + "="*50)
print("1. Внимательно проанализируйте вывод анализа df (из {}):".format(INITIAL_DATA_FILE))
print("   - Какие молекулы не прошли фильтры и почему?")
print("   - Какие основные структурные кластеры выявлены?")
print("   - Каковы диапазоны ключевых свойств (MW, LogP, SA, QED, заряды)?")
print("2. Проанализируйте результат фильтрации ChEMBL (после исправления):")
print("   - Сколько молекул осталось С УЧЕТОМ фильтра по классу? Достаточно ли для обучения?")
print("3. Начинайте активный поиск данных BDE в литературе/базах данных.")
print("4. Начинайте поиск и тестирование предобученного SMILES-генератора.")
print("5. На основе найденных данных BDE, разработайте метод estimate_bde(mol).")
print("6. Готовьте презентацию для защиты концепции, используя полученные данные и анализ.")

print("\n" + "="*50 + "\n=== Выполнение Скрипта Завершено ===\n" + "="*50)