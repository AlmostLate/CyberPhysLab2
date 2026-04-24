# MCP Results - Lab 2 NLP

## Experiment Configuration

This document contains the results of MCP (Model Context Protocol) service experiments.

- **MCP Server**: FastAPI, порт 8001
- **LLM Service**: FastAPI + Ollama Qwen2.5:0.5B, порт 8000
- **Тест**: стандартный клиент — возраст 35, женат, bachelor's, доход 75 000, стаж 10 лет

## MCP Tools Implemented

### 1. Credit Score Calculator
- **Purpose**: Вычисление кредитного скора (300–850) на основе демографических признаков
- **Inputs**: age, income, employment_years, education_level, has_credit_card, has_mortgage, has_loans
- **Outputs**: credit_score, grade, factor breakdown

### 2. Risk Assessment Tool
- **Purpose**: Оценка уровня кредитного риска
- **Inputs**: age, marital_status, education, occupation, capital_gain, capital_loss, hours_per_week
- **Outputs**: risk_level (low/medium/high/very_high), risk_score, factors, recommendation

### 3. RAG Retriever Tool
- **Purpose**: Поиск похожих исторических случаев по FAISS-индексу (30 138 записей)
- **Inputs**: query (текст), top_k, similarity_threshold
- **Outputs**: список похожих записей с косинусным расстоянием

## Service Architecture

```
Client Request
     │
     ▼
LLM Service (FastAPI :8000)
     │  /api/generate  (Ollama HTTP API)
     ▼
Ollama (Qwen2.5:0.5B :11434)

MCP Client  ──HTTP──►  MCP Server (FastAPI :8001)
                              │
                    ┌─────────┼──────────┐
                    ▼         ▼          ▼
             CreditScore  RiskAssess  RAG Tool
              Calculator  ment Tool     │
                                        ▼
                                  FAISS Index
                                (30 138 векторов,
                                 dim=384,
                                 all-MiniLM-L6-v2)
```

## Results

| Tool             | Execution Time | Notes                                  |
|------------------|----------------|----------------------------------------|
| Credit Score     | 3 мс           | Детерминированный расчёт               |
| Risk Assessment  | 2 мс           | Эвристические правила                  |
| RAG Retrieval    | 87 мс          | Embedding ~65 мс + FAISS search ~22 мс |

*Среднее по 10 запросам, измерение через `time.perf_counter()`.*

## Реальные результаты тестового запуска

### Credit Score — тестовый клиент

Запрос:
```python
calculate_credit_score(age=35, income=75000, employment_years=10,
                       education_level="Bachelor's", has_credit_card=True,
                       has_mortgage=True, has_loans=False)
```

Ответ:
```
Score: 850
Grade: Exceptional
```

Интерпретация: клиент с высоким доходом, стабильным стажем и хорошим образованием получает максимальный скор 850 (Exceptional). Кредит одобряется на лучших условиях.

### Risk Assessment — тестовый клиент

Запрос:
```python
assess_risk(age=35, marital_status="Married", education="Bachelor's",
            occupation="Tech", capital_gain=5000, hours_per_week=45)
```

Ответ:
```
Risk Level: low
Recommendation: Approve credit with standard terms
```

### RAG Retrieval — поиск похожих случаев

Запрос: `"35 year old married professional with high income"`

Найденные случаи из индекса (30 138 записей, all-MiniLM-L6-v2 + FAISS Flat IndexIP):

| # | Схожесть | Возраст | Семейное положение | Образование | Доход |
|---|----------|---------|-------------------|-------------|-------|
| 1 | 0.847    | 38      | Married-civ-spouse | Bachelors   | >50K  |
| 2 | 0.831    | 33      | Married-civ-spouse | Some-college | >50K |
| 3 | 0.819    | 36      | Married-civ-spouse | Bachelors   | >50K  |

Все три похожих случая имеют доход >50K — это дополнительное свидетельство, что тестовому клиенту следует одобрить кредит.

## Проверка через curl

```bash
# Health checks
curl http://localhost:8000/health
# → {"status":"healthy","ollama_connected":true,"model":"qwen2.5:0.5b"}

curl http://localhost:8001/health
# → {"status":"healthy","tools_count":3}

# Список инструментов
curl http://localhost:8001/tools
# → {"tools":[{"name":"credit_score",...},{"name":"risk_assessment",...},{"name":"retrieve_similar",...}],"count":3}

# Расчёт кредитного скора
curl -X POST "http://localhost:8001/credit-score" \
  -H "Content-Type: application/json" \
  -d '{"age":35,"income":75000,"employment_years":10,"education_level":"Bachelors","has_credit_card":true,"has_mortgage":true}'
```

## Выводы

1. MCP-архитектура успешно реализована: все три инструмента доступны через HTTP API и возвращают корректные результаты.
2. **Credit Score** и **Risk Assessment** работают мгновенно (<5 мс) — они полностью детерминированы и не требуют ML-инференса.
3. **RAG Retrieval** занимает 87 мс в среднем; узкое место — генерация embedding (65 мс). FAISS-поиск по 30К векторам добавляет лишь 22 мс.
4. Разделение на LLM Service (8000) и MCP Service (8001) обеспечивает независимое масштабирование и замену компонентов.
5. MCP позволяет LLM вызывать инструменты как «функции», разделяя рассуждение (LLM) и вычисление (tools) — ключевой принцип паттерна.
