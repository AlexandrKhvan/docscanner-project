# 📄 Scanner Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-brightgreen)](https://opencv.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.11x-teal)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#-license)

Документ-сканер на базе **OpenCV + FastAPI**.  
Автоматически находит границы листа, выпрямляет перспективу и отдаёт «отсканированный» результат.

---

## ✨ Возможности v0

- Поиск границ документа (Canny + контуры, выбор 4-угольника).
- Перспективная коррекция (homography).
- Режимы вывода:
  - **bw** — ч/б «скан» (адаптивный порог) — по умолчанию
  - **gray** — серый
  - **color** — цветной
- REST API (FastAPI) + простая web-страница для загрузки.

---


---

## 🚀 Установка и запуск

### Windows (PowerShell)

```powershell
# 1) Клонирование
git clone //link 
cd scanner-project

# 2) Виртуальное окружение
python -m venv .venv
. .venv\Scripts\Activate.ps1
# Если скрипты запрещены:
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 3) Зависимости
python -m pip install -r requirements.txt

# 4) Запуск API
uvicorn api.main:app --reload --port 8000
```

### Linux/MacOS
```bash
# 1) Клонирование
git clone //link
cd scanner-project

# 2) Виртуальное окружение
python -m venv .venv && source .venv/bin/activate

# 3) Зависимости
python -m pip install -r requirements.txt

# 4) Запуск API
uvicorn api.main:app --reload --port 8000
```
## После старта:

Веб-интерфейс: http://localhost:8000

Swagger UI: http://localhost:8000/docs

### Примечание: 
api/main.py монтирует web/ как статик-корень, поэтому index.html доступен сразу по /.

## Быстрый тест:
```
curl -X POST "http://localhost:8000/scan?mode=bw" \
     -H "accept: image/png" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@examples/demo.jpg" \
     --output result.png
```

Или через Python:
```python
import requests

with open("examples/demo.jpg", "rb") as f:
    r = requests.post("http://localhost:8000/scan?mode=bw", files={"file": f})
r.raise_for_status()
with open("result.png", "wb") as out:
    out.write(r.content)
print("Saved to result.png")
```

# 📄 License

MIT — см. LICENSE.