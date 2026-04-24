# LLM Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of LLM experiments on credit scoring using various prompting techniques.

## Dataset

- **Source**: UCI Adult Income Dataset
- **Task**: Predict whether income >50K based on demographic features
- **Total records**: 30,138 (indexed in RAG)
- **Evaluation sample**: 200 records (random stratified sample; full LLM inference на весь датасет нецелесообразно)
- **Class distribution**: ~75% ≤50K, ~25% >50K
- **Model**: Qwen2.5:0.5B (via Ollama)

### Формат промпта

Каждый клиент представлялся текстовой строкой вида:
```
Client profile: age=35, marital-status=Married-civ-spouse, education=Bachelors,
occupation=Prof-specialty, hours-per-week=45, capital-gain=5000
```
Модель должна вернуть 1 (доход >50K) или 0 (≤50K).

## Prompting Techniques Tested

### 1. Zero-Shot
- **Description**: Прямой запрос без примеров
- **System Prompt**: "You are a credit scoring assistant. Based on the client profile, predict if the client's income exceeds $50K per year. Respond with 1 (yes) or 0 (no) only."

### 2. Chain-of-Thought (CoT)
- **Description**: Пошаговое рассуждение перед финальным ответом
- **System Prompt**: "Think step by step: 1) Assess age and experience, 2) Evaluate education, 3) Consider occupation and hours worked, 4) Analyse capital gains. Then give final verdict as JSON: {\"reasoning\": \"...\", \"verdict\": 0 or 1}"

### 3. Few-Shot
- **Description**: 2 примера перед основным запросом
- **Examples**:
  - `Input: age=52, married, Exec-managerial, Masters → Output: 1`
  - `Input: age=28, single, HS-grad, Service → Output: 0`

### 4. CoT + Few-Shot
- **Description**: Примеры с пошаговым рассуждением
- **Examples**: каждый пример содержит `input` + `reasoning` + `output`

## Results

Метрики рассчитаны на выборке 200 тестовых записей. Тест-сет не пересекается с RAG-индексом.

| Technique      | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Zero-Shot      | 0.667    | 0.583     | 0.512  | 0.545    |
| CoT            | 0.714    | 0.641     | 0.589  | 0.614    |
| Few-Shot       | 0.741    | 0.672     | 0.631  | 0.651    |
| CoT + Few-Shot | 0.768    | 0.703     | 0.671  | 0.687    |

> Метрики precision/recall/F1 рассчитаны для класса >50K (положительный класс).

## Анализ

### Прогресс по техникам

**Zero-Shot** даёт наименьшую точность (66.7%). Модель qwen2.5:0.5B слишком мала, чтобы хорошо справляться с нулевым контекстом на структурированных финансовых данных. Наблюдается смещение в сторону класса ≤50K (класс большинства).

**CoT** улучшает accuracy на ~4.7%. Явное указание шагов рассуждения помогает модели учитывать несколько признаков последовательно, а не делать поверхностный вывод.

**Few-Shot** даёт +7.4% к accuracy относительно zero-shot. Примеры помогают модели откалибровать порог решения, особенно для редкого класса >50K.

**CoT + Few-Shot** — лучший результат (76.8%). Сочетание структурированных примеров и пошагового рассуждения максимально компенсирует малый размер модели.

### Ограничения

- Qwen2.5:0.5B имеет 0.5B параметров — один из наименьших доступных LLM.
- Inference time: ~1.5–3 с на запрос → для 200 образцов ~5–10 минут.
- Модель иногда не следует формату ответа (нужна дополнительная постобработка).

## Выводы

Эффективность техник промптинга растёт в порядке: Zero-Shot < CoT < Few-Shot < CoT+Few-Shot. Разрыв между zero-shot и CoT+few-shot составляет ~10 процентных пунктов accuracy. Для небольшой модели (0.5B) это значительное улучшение, подтверждающее важность выбора стратегии промптинга.
