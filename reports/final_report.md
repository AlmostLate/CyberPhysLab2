# Final Report: Lab 2 NLP — LLM с MCP и RAG для кредитного скоринга

## Краткое содержание

Лабораторная работа 2 по дисциплине «Кибер-физические системы». Реализован полный пайплайн автоматизации кредитного одобрения на основе связки **LLM + MCP (Model Context Protocol) + RAG (Retrieval-Augmented Generation)**. Работа выполнена на оценку «5».

## Бизнес-постановка задачи

**Контекст**: Кредитный банк — автоматизация процесса одобрения займов.

**Задача**: Система должна:
- Принимать текстовое описание клиента
- Анализировать кредитный риск с помощью ML-инструментов
- Выдавать рекомендацию по кредиту через LLM

## Датасет

| Параметр | Значение |
|----------|---------|
| **Источник** | UCI Adult Income Dataset |
| **URL** | https://archive.ics.uci.edu/dataset/2/adult |
| **Задача** | Бинарная классификация (доход >50K или ≤50K) |
| **Признаки** | 14 (демографические + профессиональные) |
| **Размер** | 30 138 записей (после очистки) |
| **Дисбаланс** | 75.1% класс ≤50K / 24.9% класс >50K |
| **Разбивка** | 80% train / 20% test |

## Метрики оценки

| Метрика | Обоснование |
|---------|-------------|
| **Accuracy** | Общая правильность кредитных решений |
| **Precision** | Минимизация ложных одобрений (плохих кредитов) |
| **Recall** | Минимизация ложных отказов (потеря хороших клиентов) |
| **F1-Score** | Баланс при дисбалансе классов |

## Архитектура

### Компоненты системы

```
┌─────────────────────────────────────────┐
│           Client / CLI                  │
│      python -m src.inference            │
└──────────────────┬──────────────────────┘
                   │
     ┌─────────────┴──────────────┐
     ▼                            ▼
┌──────────┐               ┌──────────┐
│   LLM    │               │   MCP    │
│ Service  │               │ Service  │
│ :8000    │               │ :8001    │
└────┬─────┘               └────┬─────┘
     │ /api/generate             │
     ▼                      ┌───┴────────────┐
┌──────────┐                ▼                ▼
│  Ollama  │          Credit Score      Risk Assess
│ :11434   │          Calculator        ment Tool
│Qwen2.5   │                    │
│  0.5B    │                    ▼
└──────────┘              RAG Retriever
                          │
                          ▼
                     FAISS Index
                  (30 138 векторов,
                   dim=384,
                   all-MiniLM-L6-v2)
```

### Описание компонентов

1. **LLM Service** — FastAPI-обёртка над Ollama (Qwen2.5:0.5B); поддерживает zero-shot, CoT, few-shot, CoT+few-shot.
2. **MCP Service** — сервер инструментов (кредитный скор, риск-оценка, RAG); FastAPI на порту 8001.
3. **RAG** — sentence-transformers (all-MiniLM-L6-v2) + FAISS Flat Index; индекс на 30 138 записей.
4. **ML Models** — Logistic Regression, Random Forest, Gradient Boosting из scikit-learn.

## Структура проекта

```
Lab2/
├── README.md
├── requirements.txt
├── docker-compose.yml
├── .env
├── setup.bat
├── Dockerfile.llm
├── Dockerfile.mcp
├── src/
│   ├── config.py
│   ├── inference.py
│   ├── llm/
│   │   ├── ollama_client.py
│   │   └── service.py
│   ├── mcp/
│   │   ├── client.py
│   │   ├── server.py
│   │   └── tools.py
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── indexer.py
│   │   └── retriever.py
│   ├── ml/
│   │   ├── credit_scoring.py
│   │   └── risk_analysis.py
│   └── utils/
│       └── data_loader.py
├── experiments/
│   ├── llm_results.md
│   ├── mcp_results.md
│   └── ml_results.md
└── reports/
    └── final_report.md
```

## Чеклист реализации

### Часть 1: LLM Service
- [x] Ollama сервер с Qwen2.5:0.5B
- [x] FastAPI-обёртка (порт 8000)
- [x] Zero-shot, CoT, Few-shot, CoT+Few-shot
- [x] /health, /generate, /zero-shot, /cot, /few-shot, /cot-few-shot

### Часть 2: MCP Service
- [x] Инструмент расчёта кредитного скора (300–850)
- [x] Инструмент оценки рисков (low/medium/high/very_high)
- [x] MCP Server + Client (порт 8001)
- [x] TOOL_REGISTRY с возможностью расширения

### Часть 3: RAG
- [x] Embedding через sentence-transformers (all-MiniLM-L6-v2)
- [x] FAISS Flat Index (косинусное сходство)
- [x] Индексирование 30 138 записей датасета
- [x] Retriever с top-k поиском

### Часть 4: ML Tools
- [x] Logistic Regression
- [x] Random Forest
- [x] Gradient Boosting
- [x] Инструменты риск-анализа

## Быстрый старт

### Требования
- Python 3.10+
- Ollama (установлен локально)

### Запуск

```bash
# 1. Создать и активировать виртуальное окружение
python -m venv venv
venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 2. Запустить Ollama и загрузить модель
ollama serve             # в отдельном окне (или уже запущен как служба)
ollama pull qwen2.5:0.5b

# 3. Запустить LLM Service (порт 8000)
python -m src.llm.service

# 4. Запустить MCP Service (порт 8001) — в новом терминале
python -m src.mcp.server

# 5. Запустить инференс
python -m src.inference --input "A client with high income, married, with higher education" --technique cot_few_shot
```

## Результаты экспериментов

### LLM Prompting Techniques (Qwen2.5:0.5B, 200 тестовых образцов)

| Technique      | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Zero-Shot      | 0.667    | 0.583     | 0.512  | 0.545    |
| CoT            | 0.714    | 0.641     | 0.589  | 0.614    |
| Few-Shot       | 0.741    | 0.672     | 0.631  | 0.651    |
| CoT + Few-Shot | **0.768**| **0.703** | **0.671** | **0.687** |

### ML Models (тест 6 028 записей)

| Model               | Accuracy | Precision | Recall | F1-Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.826    | 0.764     | 0.623  | 0.687    |
| Random Forest       | 0.864    | 0.812     | 0.681  | 0.741    |
| Gradient Boosting   | **0.872**| **0.831** | **0.697** | **0.758** |

### MCP Tool Performance

| Tool             | Execution Time |
|------------------|----------------|
| Credit Score     | 3 мс           |
| Risk Assessment  | 2 мс           |
| RAG Retrieval    | 87 мс          |

### RAG Quality

- Индекс: 30 138 векторов, dim=384, FAISS IndexFlatIP (cosine)
- Модель эмбеддингов: all-MiniLM-L6-v2
- Среднее косинусное сходство top-1: 0.847
- При запросе «35 year old married professional with high income» все три найденных случая имеют доход >50K

## Ключевые технические решения

1. **FAISS на Windows с кириллицей** — C++ `FileIOWriter` не поддерживает non-ASCII пути; заменено на `faiss.serialize_index()` + `open(path, 'wb')`.
2. **Ollama API** — все эндпоинты имеют префикс `/api/`: `/api/generate`, `/api/chat`, `/api/tags`.
3. **`reload=False`** в uvicorn — при запуске через `python -m` горячая перезагрузка требует app как строку-импорт, что конфликтует с прямым вызовом.
4. **Разделение `__init__.py`** — импорт `service.py` из `__init__.py` вызывал RuntimeWarning при первичном импорте модуля; убрано.

## Технологии

| Стек | Компонент |
|------|-----------|
| FastAPI + uvicorn | HTTP API |
| Ollama | Локальный LLM inference |
| sentence-transformers | Текстовые эмбеддинги |
| FAISS | Векторный поиск |
| scikit-learn | ML модели |
| pandas + numpy | Обработка данных |

## Выводы

1. **LLM vs ML**: Традиционные ML-модели превосходят LLM (0.5B) на структурированных данных (87.2% vs 76.8%). LLM полезен для неструктурированных входных данных и объяснимости решений.
2. **CoT+Few-Shot** — наиболее эффективная техника промптинга: +10.1 п.п. accuracy относительно zero-shot.
3. **RAG** обогащает контекст LLM реальными историческими случаями — помогает обосновать решение.
4. **MCP-архитектура** позволяет LLM использовать специализированные инструменты как «функции» — разделение рассуждения и вычисления.
5. **Gradient Boosting** — лучшая модель для продакшена; рекомендуется добавить class_weight='balanced' для улучшения Recall.

## Автор

Лабораторная работа 2, дисциплина «Кибер-физические системы»

## Дата

24 апреля 2026 г.
