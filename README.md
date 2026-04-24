# Лабораторная работа 2: NLP — LLM + MCP + RAG для кредитного скоринга

Лабораторная работа по курсу «Киберфизические системы». Тема — применение LLM к задаче автоматизации одобрения кредитов, вариант на пятёрку.

## Бизнес-задача

Заказчик — кредитный банк. Нужно сделать прототип, который принимает словесное описание клиента и решает, одобрять ли кредит. Используется LLM (Qwen2.5:0.5B через Ollama) + инструменты классического ML через MCP + RAG для поиска похожих исторических случаев.

Датасет: UCI Adult Income Dataset — данные переписи населения США, целевая переменная — доход >50K или ≤50K (прокси для кредитоспособности).

## Архитектура

Три слоя:

1. **LLM-сервис** — FastAPI обёртка вокруг Ollama с Qwen2.5:0.5B. Поддерживает zero-shot, CoT, few-shot и CoT+few-shot промптинг.
2. **MCP-сервис** — FastAPI с набором инструментов: расчёт кредитного скора, оценка риска по демографическим данным, RAG-ретривер для поиска похожих случаев.
3. **ML-инструменты** — LogisticRegression, RandomForest, GradientBoosting для точного скоринга, встроены как MCP-тулы.

## Структура проекта

```
Lab2/
├── README.md
├── requirements.txt
├── setup.bat
├── docker-compose.yml      # поднимает ollama + llm_service + mcp_service
├── Dockerfile.llm          # образ для LLM-сервиса
├── Dockerfile.mcp          # образ для MCP-сервиса
├── .env                    # переменные окружения (порты, URL)
├── src/
│   ├── config.py           # все настройки
│   ├── inference.py        # скрипт для тестирования
│   ├── llm/
│   │   ├── ollama_client.py    # клиент Ollama API
│   │   └── service.py          # FastAPI LLM-сервис
│   ├── mcp/
│   │   ├── server.py           # MCP-сервер (FastAPI)
│   │   ├── client.py           # MCP-клиент
│   │   └── tools.py            # инструменты: кредит, риск, RAG
│   ├── rag/
│   │   ├── embedder.py         # эмбеддинги через sentence-transformers
│   │   ├── indexer.py          # FAISS-индекс
│   │   └── retriever.py        # поиск похожих случаев
│   ├── ml/
│   │   ├── credit_scoring.py   # ML-модели скоринга
│   │   └── risk_analysis.py    # анализ риска
│   └── utils/
│       └── data_loader.py      # загрузка и препроцессинг датасета
├── experiments/
│   ├── llm_results.md
│   ├── mcp_results.md
│   └── ml_results.md
└── reports/
    └── final_report.md
```

## Установка и запуск

### Требования
- Python 3.8+
- Docker + Docker Compose (для запуска через контейнеры)
- Либо Ollama установленный локально (для запуска без Docker)

---

### Вариант A: запуск через Docker (рекомендуется)

#### Шаг 1 — Установить зависимости Python

```bash
setup.bat
```

Или вручную:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

#### Шаг 2 — Поднять контейнеры

```bash
docker compose up -d
```

Это запустит три сервиса:
- `lab2_ollama` — Ollama на порту **11434**
- `lab2_llm_service` — LLM FastAPI на порту **8000**
- `lab2_mcp_service` — MCP FastAPI на порту **8001**

Проверить, что всё запустилось:
```bash
docker ps
```

#### Шаг 3 — Скачать модель в контейнер

```bash
docker exec lab2_ollama ollama pull qwen2.5:0.5b
```

Модель весит ~350 МБ, скачается один раз и сохранится в volume `ollama_data`.

#### Шаг 4 — Проверить работу сервисов

```bash
curl http://localhost:8000/health
curl http://localhost:8001/health
curl http://localhost:8001/tools
```

Если всё ок — в ответе будет `{"status": "healthy", ...}`.

#### Шаг 5 — Проиндексировать датасет для RAG

Датасет Adult Income скачать с [UCI](https://archive.ics.uci.edu/dataset/2/adult) и положить в `data/adult.csv`, затем:

```bash
python src/utils/data_loader.py
```

#### Шаг 6 — Запустить инференс

```bash
# из виртуального окружения (venv), запросы идут к контейнерам
python src/inference.py --input "Клиент 35 лет, женат, высшее образование, работает в IT, доход 75000 в год"
```

---

### Вариант B: запуск локально (без Docker)

#### Шаг 1 — Установить зависимости

```bash
setup.bat
venv\Scripts\activate
```

#### Шаг 2 — Установить и запустить Ollama

Скачать с [ollama.com](https://ollama.com), затем:
```bash
ollama serve
```

В отдельном терминале скачать модель:
```bash
ollama pull qwen2.5:0.5b
```

Проверить:
```bash
curl http://localhost:11434/api/tags
```

#### Шаг 3 — Запустить LLM-сервис

В новом терминале (с активированным venv):
```bash
python src/llm/service.py
```

Сервис поднимется на `http://localhost:8000`. Документация API: `http://localhost:8000/docs`.

#### Шаг 4 — Запустить MCP-сервис

Ещё один терминал:
```bash
python src/mcp/server.py
```

Сервис поднимется на `http://localhost:8001`. Список инструментов: `http://localhost:8001/tools`.

#### Шаг 5 — Проиндексировать датасет и запустить инференс

```bash
python src/utils/data_loader.py
python src/inference.py --input "Клиент 45 лет, женат, образование Master's, работает в Finance"
```

---

### Примеры запросов через curl

Прямой запрос к LLM (zero-shot):
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d "{\"prompt\": \"Client: 35 years old, married, Bachelor degree, income 60000. Approve credit?\", \"max_tokens\": 200}"
```

Запрос к MCP — рассчитать кредитный скор:
```bash
curl -X POST http://localhost:8001/tools/execute \
  -H "Content-Type: application/json" \
  -d "{\"tool_name\": \"calculate_credit_score\", \"arguments\": {\"age\": 35, \"income\": 75000, \"employment_years\": 10, \"education_level\": \"Bachelor's\"}}"
```

Запрос к MCP — оценить риск:
```bash
curl -X POST http://localhost:8001/tools/execute \
  -H "Content-Type: application/json" \
  -d "{\"tool_name\": \"assess_risk\", \"arguments\": {\"age\": 35, \"marital_status\": \"Married\", \"education\": \"Bachelor's\", \"occupation\": \"Tech\"}}"
```

## Промптинг-техники

Реализованы и протестированы на задаче определения дохода >50K:

- **Zero-shot** — просто описание задачи без примеров
- **CoT** — шаг за шагом: анализ демографии → финансы → занятость → вывод
- **Few-shot** — несколько примеров с ответами в промпте
- **CoT + Few-shot** — примеры с рассуждениями

Формат вывода LLM — JSON: `{"reasoning": "...", "verdict": 0/1}`.

## Метрики оценки

| Техника | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|----|
| Zero-shot | — | — | — | — |
| CoT | — | — | — | — |
| Few-shot | — | — | — | — |
| CoT + Few-shot | — | — | — | — |
| ML (Random Forest) | — | — | — | — |

Результаты заполняются после прогона экспериментов.

## Стек

- FastAPI + uvicorn
- Ollama (Qwen2.5:0.5B)
- FastMCP
- sentence-transformers + FAISS
- scikit-learn
- Docker + Docker Compose
