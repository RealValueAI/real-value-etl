# RealValue ETL Service

ETL-сервис для переноса данных из S3 в ClickHouse, включая проверку, трансформацию, 
маппинг данных из разных источников, валидацию, а также REST API для управления 
процессом с использованием FastAPI.

## Функциональность
- Обработка данных из S3 с настройкой платформ и дат.
- Трансформация 
- Загрузка данных в ClickHouse.
- REST API для управления процессом.


### Клонирование репозитория
```bash
git clone https://github.com/your-repo/realvalue-etl-service.git
cd real-value-etl
```
### Зависимости
```bash
poetry install
```
### Запуск
```bash
uvicorn main:app --port 8100 --reload
```

### API Эндпоинты
GET /
Приветственный эндпоинт.

POST /etl/start
Запускает ETL-процесс.

Пример тела запроса:
```json
{
  "platforms": {
    "domclick": "latest",
    "yandex": "20241208",
    "cian": "skip",
    "avito": null
  }
}
