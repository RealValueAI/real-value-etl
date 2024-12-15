import numpy as np
from clickhouse_connect import get_client
import pandas as pd
from src.utils.logger import get_logger
from abc import ABC, abstractmethod


class BaseLoader(ABC):
    @abstractmethod
    def load(self, df: pd.DataFrame, table_name: str):
        pass


class ClickHouseLoader(BaseLoader):
    def __init__(self, config):
        self.client = get_client(
            host=config.host,
            port=config.port,
            username=config.user,
            password=config.password,
            database=config.database,
            connect_timeout=90,
        )
        self.logger = get_logger(self.__class__.__name__.lower())

    def load(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            self.logger.info("DataFrame is empty. Nothing to load.")
            return

        try:
            # Очищаем таблицу перед загрузкой
            self.logger.info(
                f"Truncating table {table_name} before loading..."
                )
            self.client.command(f"TRUNCATE TABLE {table_name}")
            self.logger.info(f"Table {table_name} truncated successfully.")
            chunk_size = 50000
            # Разбиваем DataFrame на чанки
            chunks = np.array_split(df, max(len(df) // chunk_size, 1))

            for i, chunk in enumerate(chunks):
                self.logger.info(
                    f"Loading chunk {i + 1} of {len(chunks)} with {len(chunk)} rows..."
                    )
                self.client.insert_df(table_name, chunk)

            self.logger.info(
                f"Successfully loaded data into {table_name} (ClickHouse). Total rows: {len(df)}"
                )
        except Exception as e:
            self.logger.error(f"Failed to load data into ClickHouse: {e}")
            raise


class CSVLoader(BaseLoader):
    def __init__(self, output_dir: str = "./output"):
        self.output_dir = output_dir
        self.logger = get_logger(self.__class__.__name__.lower())

    def load(self, df: pd.DataFrame, table_name: str):
        if df.empty:
            self.logger.info("DataFrame is empty. Nothing to load.")
            return

        try:
            # Сохраняем данные в локальный CSV файл
            file_name = f"{self.output_dir}/{table_name}.csv"
            df.to_csv(file_name, index=False)
            self.logger.info(
                f"Successfully saved data into {file_name} (CSV mode)"
                )
        except Exception as e:
            self.logger.error(f"Failed to save data to CSV: {e}")
            raise
