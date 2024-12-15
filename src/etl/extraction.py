from abc import ABC, abstractmethod
import boto3
import pandas as pd
from io import StringIO

from pandas import DataFrame

from src.utils.logger import get_logger


class BaseExtractor(ABC):
    @abstractmethod
    def extract(self, source: str) -> pd.DataFrame:
        pass


class S3Extractor(BaseExtractor):
    def __init__(self, config):
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=config.access_key,
            aws_secret_access_key=config.secret_key
        )
        self.bucket_name = config.bucket_name
        self.folder_name = config.folder_name

    def extract(self, file_path: str) -> pd.DataFrame | None:
        # Загрузка файла из S3
        full_key = f"{self.folder_name}{file_path}"
        obj = self.s3.get_object(Bucket=self.bucket_name, Key=full_key)
        data = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(data))
        return df


class DomClickExtractor(S3Extractor):
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger('domclick_extractor_logger')
        self.logger.info("DomClickExtractor initialized.")

    def extract(self, file_path: str) -> DataFrame | None:
        full_key = f"{self.folder_name}{file_path}"
        try:
            # Загрузка файла из S3
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=full_key)
            data = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
            self.logger.info(f"Successfully extracted data for: {full_key}")
            return df
        except self.s3.exceptions.NoSuchKey:
            self.logger.info(f"No data available: {full_key}")
            return None


class YandexExtractor(S3Extractor):
    def __init__(self, config):
        super().__init__(config)
        self.logger = get_logger('yandex_extractor_logger')
        self.logger.info("YandexExtractor initialized.")

    def extract(self, file_path: str) -> DataFrame | None:
        full_key = f"{self.folder_name}{file_path}"
        try:
            # Загрузка файла из S3
            obj = self.s3.get_object(Bucket=self.bucket_name, Key=full_key)
            data = obj['Body'].read().decode('utf-8')
            df = pd.read_csv(StringIO(data))
            self.logger.info(f"Successfully extracted data for: {full_key}")
            return df
        except self.s3.exceptions.NoSuchKey:
            self.logger.info(f"No data available: {full_key}")
            return None
