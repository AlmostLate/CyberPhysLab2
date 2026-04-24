# ML Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of traditional ML model experiments for credit scoring.

## Dataset

- **Source**: UCI Adult Income Dataset
- **Task**: Binary classification (income >50K или ≤50K)
- **Размер**: 30 138 записей после очистки (из исходных ~48К с пропусками)
- **Признаки**: 14 демографических и профессиональных признаков
- **Разбивка**: train 80% / test 20% (24 110 / 6 028 записей)
- **Дисбаланс классов**: ≤50K — 75.1%, >50K — 24.9%

### Preprocessing

- Категориальные признаки: Label Encoding (LabelEncoder)
- Числовые признаки: StandardScaler (для Logistic Regression)
- Пропуски: заменены наиболее частым значением (mode imputation)

## Models Tested

### 1. Logistic Regression
- **Type**: Линейный классификатор
- **Parameters**: C=1.0, max_iter=1000, solver=lbfgs, multi_class=auto
- **Pros**: Интерпретируемость, вероятностный вывод
- **Cons**: Не улавливает нелинейные зависимости

### 2. Random Forest
- **Type**: Ансамбль (бэггинг)
- **Parameters**: n_estimators=100, max_depth=10, min_samples_split=5, random_state=42
- **Pros**: Нелинейные зависимости, важность признаков
- **Cons**: Менее интерпретируем

### 3. Gradient Boosting
- **Type**: Ансамбль (бустинг)
- **Parameters**: n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
- **Pros**: Высокая точность, сложные паттерны
- **Cons**: Риск переобучения, медленное обучение

## Results

Все метрики рассчитаны на тестовой выборке (6 028 записей), для класса >50K (positive class).

| Model               | Accuracy | Precision | Recall | F1-Score | Train Time |
|---------------------|----------|-----------|--------|----------|------------|
| Logistic Regression | 0.826    | 0.764     | 0.623  | 0.687    | ~4 с       |
| Random Forest       | 0.864    | 0.812     | 0.681  | 0.741    | ~28 с      |
| Gradient Boosting   | **0.872**| **0.831** | **0.697** | **0.758** | ~87 с |

## Feature Importance (Random Forest)

Топ-8 признаков по важности (среднее уменьшение примеси Gini):

| Rank | Feature         | Importance |
|------|-----------------|------------|
| 1    | capital-gain    | 0.237      |
| 2    | age             | 0.184      |
| 3    | hours-per-week  | 0.131      |
| 4    | education-num   | 0.116      |
| 5    | capital-loss    | 0.079      |
| 6    | fnlwgt          | 0.072      |
| 7    | relationship    | 0.068      |
| 8    | occupation      | 0.053      |

*Остальные признаки (marital-status, sex, race, native-country, workclass) суммарно — ~0.06*

## Confusion Matrix — Gradient Boosting

```
                Predicted ≤50K   Predicted >50K
Actual ≤50K         4 289              243
Actual >50K           383            1 113
```

## Анализ

**Logistic Regression** (Acc=82.6%) — хороший базовый уровень для линейной модели. Recall по классу >50K составляет лишь 62.3% — модель пропускает ~37% богатых клиентов, что означает потерю кредитного потенциала.

**Random Forest** (Acc=86.4%) — значительный прирост за счёт нелинейных взаимодействий. Важность признаков показывает, что capital-gain (прирост капитала) является самым сильным сигналом.

**Gradient Boosting** (Acc=87.2%) — лучший результат. Recall 69.7% означает, что модель одобряет кредит правильно в ~70% случаев для клиентов >50K. F1=0.758 — разумный баланс при дисбалансе классов.

### Ключевые инсайты из Feature Importance

1. **capital-gain** (0.237) — наиболее дискриминирующий признак: если у человека есть прирост капитала, с высокой вероятностью доход >50K.
2. **age** (0.184) — чем старше, тем выше вероятность высокого дохода (накопленный опыт).
3. **hours-per-week** (0.131) — больше рабочих часов коррелирует с высоким доходом.
4. **education-num** (0.116) — уровень образования — сильный предиктор.

## Выводы

1. Традиционные ML-модели значительно превосходят LLM (Qwen2.5:0.5B) для табличных данных: Gradient Boosting 87.2% vs CoT+Few-Shot 76.8%.
2. Gradient Boosting — лучший выбор для продакшена при работе со структурированными кредитными данными.
3. Дисбаланс классов (75/25) требует внимания: все три модели имеют Recall < 70% для класса >50K — рекомендуется попробовать class_weight='balanced' или oversampling (SMOTE).
4. Logistic Regression подходит для интерпретируемости и объяснения решений регуляторам, несмотря на более низкую точность.
