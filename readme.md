# ML Chemistry: Machine Learning для молекулярного дизайна и оптимизации

Комплексный проект по применению машинного обучения, больших языковых моделей (LLM) и генетических алгоритмов для генерации, оптимизации и анализа химических соединений.

## Содержание

- [О проекте](#о-проекте)
- [Возможности](#возможности)
- [Структура проекта](#структура-проекта)
- [Установка](#установка)
- [Использование](#использование)
- [Датасеты и модели](#датасеты-и-модели)
- [Описание ноутбуков](#описание-ноутбуков)
- [Технологический стек](#технологический-стек)
- [Лицензия](#лицензия)

## О проекте

ML Chemistry представляет собой исследовательскую платформу, объединяющую современные подходы машинного обучения и хемоинформатики для решения задач молекулярного дизайна. Проект включает в себя:

- **Генерацию молекул** с использованием transformer-based моделей и химических LLM
- **Оптимизацию структур** через генетические алгоритмы с граф-представлением молекул
- **Предсказание свойств** с помощью классических ML и deep learning моделей
- **Ретросинтетический анализ** для планирования синтеза целевых соединений

##  Возможности

-  **Exploratory Data Analysis (EDA)** химических датасетов
-  **Inference с химическими LLM** (ChemDFM, ChemLLM, Galactica)
-  **Предсказание физико-химических свойств** (PDSC, BDE)
-  **Генетические алгоритмы** для молекулярной оптимизации
-  **Граф-based представление** молекул для эволюционных алгоритмов
-  **Ретросинтетическое планирование** химических реакций

##  Структура проекта

```
ml-chemistry/
│
├── 1_EDA.ipynb                              # Разведочный анализ данных
├── 2_mol_gpt_base_inference.ipynb          # Inference с MolGPT
├── 3_chem_llms_inference.ipynb             # Работа с химическими LLM
├── 4_ml_PDSC_prediction.ipynb              # Предсказание PDSC методами ML
├── 5_rl_genetic_algo.ipynb                 # RL и генетические алгоритмы
├── 6_genetic_algo_graph_based.ipynb        # Граф-based генетический алгоритм
├── 7_genetic_algo_graph_based_final.ipynb  # Финальная версия GA
│
├── llm_interface.py                         # Интерфейс для работы с LLM
├── otbor.py                                 # Модуль отбора для GA
├── retro.py                                 # Ретросинтетический анализ
├── utils.py                                 # Вспомогательные функции
│
├── population.csv                           # Популяция молекул
├── population_mol_based.csv                 # Молекулярная популяция
├── smiles_for_rxn.txt                       # SMILES для реакций
│
├── requirements.txt                         # Python зависимости
├── bde_lib_requirements.txt                # Зависимости для расчета BDE
└── LICENSE                                  # Apache 2.0 License
```

##  Установка

### Предварительные требования

- Python 3.8 или выше
- CUDA-compatible GPU (рекомендуется для inference с LLM)
- Минимум 16GB RAM (32GB+ рекомендуется для больших моделей)

### Шаги установки

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Kali-Uga/ml-chemistry.git
cd ml-chemistry
```

2. Создайте виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate  # для Linux/macOS
# или
venv\Scripts\activate     # для Windows
```

3. Установите основные зависимости:
```bash
pip install -r requirements.txt
```

4. Для работы с расчетом BDE (Bond Dissociation Energy):
```bash
pip install -r bde_lib_requirements.txt
```

5. Запустите Jupyter:
```bash
jupyter notebook
```

##  Использование

### Базовый workflow

1. **Анализ данных**: Начните с `1_EDA.ipynb` для понимания структуры датасета
2. **Генерация молекул**: Используйте `2_mol_gpt_base_inference.ipynb` или `3_chem_llms_inference.ipynb`
3. **Предсказание свойств**: Примените `4_ml_PDSC_prediction.ipynb` для оценки характеристик
4. **Оптимизация**: Запустите `7_genetic_algo_graph_based_final.ipynb` для поиска оптимальных структур

### Пример использования LLM интерфейса

```python
from llm_interface import ChemLLM

# Инициализация модели
model = ChemLLM(model_name="AI4Chem/ChemLLM-7B-Chat")

# Генерация молекулы
prompt = "Generate a molecule with high binding affinity to protein target X"
generated_smiles = model.generate(prompt)

print(f"Generated SMILES: {generated_smiles}")
```

### Пример генетического алгоритма

```python
from otbor import GeneticOptimizer
from utils import calculate_fitness

# Инициализация популяции
optimizer = GeneticOptimizer(
    population_size=100,
    mutation_rate=0.1,
    crossover_rate=0.7
)

# Запуск оптимизации
best_molecule = optimizer.evolve(
    generations=50,
    fitness_function=calculate_fitness
)
```

##  Датасеты и модели

### Используемые датасеты

- **ZINC15** ([jonghyunlee/ZINC15](https://huggingface.co/datasets/jonghyunlee/ZINC15)) - крупномасштабная база данных коммерчески доступных соединений
- **ChEMBL** - база биоактивных молекул с данными о лекарственных свойствах
- **SMILES Transformers** ([maykcaldas/smiles-transformers](https://huggingface.co/datasets/maykcaldas/smiles-transformers)) - датасет для обучения transformer моделей
- **BDE датасеты** - данные о энергиях диссоциации связей

### Используемые модели

| Модель | Параметры | Назначение |
|--------|-----------|------------|
| [OpenDFM/ChemDFM-v1.0-13B](https://huggingface.co/OpenDFM/ChemDFM-v1.0-13B) | 13B | Универсальная химическая foundation модель |
| [AI4Chem/ChemLLM-7B-Chat](https://huggingface.co/AI4Chem/ChemLLM-7B-Chat) | 7B | Специализированная химическая chat-модель |
| [facebook/galactica-6.7b](https://huggingface.co/facebook/galactica-6.7b) | 6.7B | Научная языковая модель |

##  Описание ноутбуков

### 1️ `1_EDA.ipynb` - Разведочный анализ данных
Комплексный анализ химических датасетов:
- Визуализация распределений молекулярных дескрипторов
- Статистический анализ физико-химических свойств
- Проверка качества и чистоты данных
- Идентификация паттернов и корреляций

### 2️ `2_mol_gpt_base_inference.ipynb` - MolGPT Inference
Работа с базовой моделью MolGPT:
- Загрузка и инициализация предобученной модели
- Генерация новых молекулярных структур
- Оценка валидности и уникальности генераций
- Benchmark производительности модели

### 3️ `3_chem_llms_inference.ipynb` - Химические LLM
Inference с крупными языковыми моделями:
- Сравнение ChemDFM, ChemLLM и Galactica
- Промпт-инжиниринг для химических задач
- Многозадачное обучение и zero-shot learning
- Оценка качества генерации

### 4️ `4_ml_PDSC_prediction.ipynb` - Предсказание свойств
Machine learning для предсказания физико-химических характеристик:
- Feature engineering молекулярных дескрипторов
- Обучение регрессионных моделей (RF, XGBoost, Neural Networks)
- Кросс-валидация и оценка метрик
- Интерпретация важности признаков

### 5️ `5_rl_genetic_algo.ipynb` - RL и генетические алгоритмы
Reinforcement Learning подход к молекулярной оптимизации:
- Формулировка reward function
- Реализация policy gradient методов
- Сравнение RL с классическими GA
- Анализ сходимости и эффективности

### 6️ `6_genetic_algo_graph_based.ipynb` - Граф-based GA
Генетический алгоритм с граф-представлением:
- Представление молекул в виде графов
- Граф-based операторы мутации и кроссовера
- Сохранение валидности структур
- Визуализация эволюции популяции

### 7️ `7_genetic_algo_graph_based_final.ipynb` - Финальная версия GA
Оптимизированная и производственно-готовая версия:
- Улучшенные операторы эволюции
- Адаптивные параметры GA
- Многокритериальная оптимизация (NSGA-II)
- Экспорт оптимизированных структур

##  Технологический стек

### Core Libraries
- **PyTorch** / **TensorFlow** - Deep Learning frameworks
- **Transformers (Hugging Face)** - Работа с LLM
- **RDKit** - Хемоинформатика и молекулярные манипуляции
- **scikit-learn** - Классические ML алгоритмы

### Визуализация и анализ
- **Matplotlib** / **Seaborn** - Построение графиков
- **py3Dmol** - 3D визуализация молекул
- **NetworkX** - Граф-анализ

### Специализированные библиотеки
- **DeepChem** - Deep learning для химии
- **OpenBabel** - Конверсия химических форматов
- **SELFIES** - Альтернативное представление молекул

##  Контрибьюция

Вклад в проект приветствуется! Пожалуйста:

1. Форкните репозиторий
2. Создайте feature branch (`git checkout -b feature/AmazingFeature`)
3. Закоммитьте изменения (`git commit -m 'Add some AmazingFeature'`)
4. Запушьте в branch (`git push origin feature/AmazingFeature`)
5. Откройте Pull Request

##  Roadmap

- [ ] Добавление web-интерфейса для генерации молекул
- [ ] Интеграция с квантово-химическими пакетами (Gaussian, ORCA)
- [ ] Реализация active learning для оптимизации обучения
- [ ] Поддержка многоагентных систем для collaborative design
- [ ] Расширение поддержки различных типов химических реакций

##  Лицензия

Проект распространяется под лицензией Apache License 2.0. См. файл [LICENSE](LICENSE) для деталей.

##  Цитирование

Если вы используете этот проект в своих исследованиях, пожалуйста, укажите ссылку:

```bibtex
@misc{ml-chemistry2024,
  author = {Nikita (Kali-Uga) & Vladimir},
  title = {ML Chemistry: Machine Learning для молекулярного дизайна},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/Kali-Uga/ml-chemistry}
}
```

##  Полезные ресурсы

- [RDKit Documentation](https://www.rdkit.org/docs/)
- [DeepChem Tutorials](https://deepchem.io/)
- [Molecular Generation Review Paper](https://arxiv.org/abs/2203.04119)
- [SMILES Tutorial](https://www.daylight.com/dayhtml/doc/theory/theory.smiles.html)

##  Disclaimer

Этот проект предназначен исключительно для исследовательских и образовательных целей. Сгенерированные молекулярные структуры требуют экспериментальной валидации перед практическим применением. Авторы не несут ответственности за использование результатов в коммерческих или медицинских целях без должной проверки.

---

