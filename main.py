import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional, Dict
from src.etl.datapipeline import DataPipeline
from src.utils.config import PLATFORMS, s3_config, clickhouse_config
from src.utils.checking_s3_data import PlatformsDateResolver


class PlatformRequest(BaseModel):
    """
    Ключ - имя платформы, значение - либо дата (строка),
    либо "latest", либо "skip", либо None
    например: {"domclick": "latest", "yandex": "20241208"}
    """
    platforms: Dict[str, Optional[str]] = {
        'domclick': 'latest',
        'yandex': 'latest',
        'cian': 'skip',
        'avito': 'skip',
    }


app = FastAPI()


@app.get("/")
async def root():
    return {
        "message": "Welcome to RealValue ETL Service. Take a look at docs."
    }


@app.post("/etl/start")
async def start_etl(request: PlatformRequest):
    """
    Пример тела запроса:
    {
        "platforms": {
            "domclick": "latest",
            "yandex": "skip",
            "cian": "20241201"
        }
    }

    Логика:
    - Если значение "latest", пытаемся найти последнюю доступную дату для данной платформы.
    - Если значение "skip", пропускаем эту платформу.
    - Если указана дата в формате YYYYMMDD, берём именно эту дату.
    - Если None или отсутствует платформа, можно либо пропустить, либо также попытаться "latest".
    """

    resolver = PlatformsDateResolver(
        platforms=PLATFORMS,
        s3_config=s3_config,
    )
    final_dates = resolver.resolve_dates(request_body=request.platforms)

    pipeline = DataPipeline(
        s3_config=s3_config,
        clickhouse_config=clickhouse_config,
        platform_date_map=final_dates,
    )
    result = pipeline.run(mode='production')
    return result


if __name__ == "__main__":
    uvicorn.run("main:app", port=8100, reload=True)
