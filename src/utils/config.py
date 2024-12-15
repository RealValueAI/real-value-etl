import os
from dataclasses import dataclass
from datetime import datetime
from typing import List

from dotenv import load_dotenv

PLATFORMS: List[str] = ['domclick', 'yandex', 'cian', 'avito']

TODAY_STR = datetime.now().strftime('%Y%m%d')

DOMCLICK_S3_STR = 'offers_data/domclick_'
DOMCLICK_FORMAT = '.csv'

YANDEX_S3_STR = 'offers_data/yandex_'
YANDEX_FORMAT = '.csv'
load_dotenv()


@dataclass
class S3Config:
    access_key: str = os.getenv("YANDEX_CLOUD_ACCESS_KEY")
    secret_key: str = os.getenv("YANDEX_CLOUD_SECRET_KEY")
    bucket_name: str = os.getenv("YANDEX_BUCKET_NAME")
    folder_name: str = 'offers_data/'


@dataclass
class ClickHouseConfig:
    host: str = os.getenv("CLICKHOUSE_HOST")
    port: int = int(os.getenv("CLICKHOUSE_PORT"))
    user: str = os.getenv("CLICKHOUSE_USER")
    password: str = os.getenv("CLICKHOUSE_PASSWORD")
    database: str = os.getenv("CLICKHOUSE_DATABASE")
    table_name: str = 'listings_ml_vitrine'


s3_config = S3Config()
clickhouse_config = ClickHouseConfig()
