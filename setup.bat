@echo off
REM Lab 2 NLP - Setup Script for Windows

echo ============================================
echo Lab 2 NLP - Environment Setup (Windows)
echo ============================================
echo.

REM --- Python ---
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python не найден.
    echo Установи Python 3.8+ с https://www.python.org/downloads/
    echo Убедись что при установке поставлена галочка "Add Python to PATH"
    pause
    exit /b 1
)
echo [OK] Python:
python --version

REM --- Docker ---
echo.
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Docker не найден. Вариант с контейнерами недоступен.
    echo     Чтобы использовать Docker, установи Docker Desktop:
    echo     https://www.docker.com/products/docker-desktop/
    echo     После установки перезагрузи компьютер и запусти скрипт снова.
    set DOCKER_FOUND=0
) else (
    echo [OK] Docker:
    docker --version
    set DOCKER_FOUND=1
)

REM --- Ollama ---
echo.
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Ollama не найдена. Для локального запуска без Docker нужна Ollama.
    echo     Скачай и установи: https://ollama.com/download
    echo     После установки выполни: ollama pull qwen2.5:0.5b
    set OLLAMA_FOUND=0
) else (
    echo [OK] Ollama:
    ollama --version
    set OLLAMA_FOUND=1
)

REM --- Виртуальное окружение ---
echo.
if not exist "venv" (
    echo [1/3] Создаю виртуальное окружение...
    python -m venv venv
) else (
    echo [1/3] Виртуальное окружение уже есть, пропускаю...
)

echo [2/3] Активирую виртуальное окружение...
call venv\Scripts\activate.bat

echo [3/3] Устанавливаю зависимости Python...
python -m pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo [OK] Зависимости установлены.

REM --- Итог ---
echo.
echo ============================================
echo Установка завершена!
echo ============================================
echo.

if "%DOCKER_FOUND%"=="1" (
    echo Вариант A - через Docker:
    echo   docker compose up -d
    echo   docker exec lab2_ollama ollama pull qwen2.5:0.5b
    echo.
)

if "%OLLAMA_FOUND%"=="1" (
    echo Вариант B - локально ^(Ollama уже установлена^):
    echo   ollama serve                      ^<-- в отдельном терминале
    echo   ollama pull qwen2.5:0.5b
    echo   python -m src.llm.service         ^<-- ещё один терминал
    echo   python -m src.mcp.server          ^<-- ещё один терминал
    echo.
)

if "%DOCKER_FOUND%"=="0" if "%OLLAMA_FOUND%"=="0" (
    echo [!] Ни Docker, ни Ollama не найдены.
    echo     Нужно установить хотя бы одно из двух:
    echo.
    echo     Ollama ^(проще^):  https://ollama.com/download
    echo     Docker Desktop:   https://www.docker.com/products/docker-desktop/
    echo.
)

echo Активировать окружение в следующий раз:
echo   call venv\Scripts\activate.bat
echo ============================================
pause
