from typing import Dict, Literal, Optional

import pandas as pd

from src.etl.extraction import AvitoExtractor, CianExtractor, \
    DomClickExtractor, S3Extractor, \
    YandexExtractor
from src.etl.merging import UnifiedDataMerger
from src.etl.transformation import (
    AvitoTransformer, BaseTransformer,
    DomClickTransformer,
    YandexTransformer,
)
from src.etl.loading import CSVLoader, ClickHouseLoader
from src.utils.config import (
    S3Config,
    ClickHouseConfig,
)
from src.utils.logger import get_logger
from src.utils.types_transform import transform_to_clickhouse_schema


class DataPipeline:
    def __init__(
        self,
        s3_config: S3Config,
        clickhouse_config: ClickHouseConfig,
        platform_date_map: Dict[str, Optional[str]]
    ):
        self.s3_config = s3_config
        self.clickhouse_config = clickhouse_config
        self.platform_date_map = platform_date_map
        self.logger = get_logger('data_pipeline_logger')

        # Экстракторы для всех платформ
        self.extractors = {
            "domclick": DomClickExtractor(self.s3_config),
            "yandex": YandexExtractor(self.s3_config),
            "cian": CianExtractor(self.s3_config),
            "avito": AvitoExtractor(self.s3_config),
        }

        # Трансформеры для всех платформ
        self.transformers = {
            "domclick": DomClickTransformer(),
            "yandex": YandexTransformer(),
            "cian": BaseTransformer(),
            "avito": AvitoTransformer(),
        }

        # Мерджер под формат dwh
        self.merger = UnifiedDataMerger()

        # лоадеры
        self.loaders = {
            "production": ClickHouseLoader(self.clickhouse_config),
            "test": CSVLoader(),
        }

    def run(self, mode: Literal['production', 'test'] = 'production'):
        dataframes = {}
        if mode == 'production':
            for platform, date_str in self.platform_date_map.items():
                # Если дата None, пропускаем платформу
                if date_str is None:
                    self.logger.info(
                        f"No data date provided for {platform}. Skipping."
                        )
                    continue

                file_name = f"{platform}_{date_str}.csv"
                extractor = self.extractors.get(platform)
                if not extractor:
                    self.logger.warning(
                        f"No extractor found for platform {platform},"
                        f" skipping."
                        )
                    continue

                df = extractor.extract(file_name)
                if df is not None and not df.empty:
                    self.logger.info(
                        f"Data extracted for {platform} - {file_name}:"
                        f" {len(df)} rows."
                        )
                    dataframes[platform] = df
                else:
                    self.logger.info(
                        f"No data returned for {platform} - {file_name},"
                        f" skipping this platform."
                        )
        elif mode == 'test':
            dataframes['domclick'] = pd.read_csv(
                'output/domclick_20241214.csv',
                nrows=5000
            )
            dataframes['yandex'] = pd.read_csv(
                'output/yandex_20241208.csv',
                nrows=5000
            )
            dataframes['avito'] = pd.read_csv(
                'output/avito_20250319.csv',
                nrows=5000
            )
            dataframes['cian'] = pd.read_csv(
                'output/cian_20241107.csv',
                nrows=5000
            )
        if not dataframes:
            self.logger.warning(
                "No data extracted for any platform. Nothing to process."
            )
            return {
                "status": "no_data",
                "message": "No platforms returned data."
            }

        transformed_data = []
        for platform, df in dataframes.items():
            transformer = self.transformers.get(platform)
            if not transformer:
                self.logger.warning(
                    f"No transformer found for {platform}, skipping transformation."
                    )
                transformed_data.append(df)
                continue
            try:
                df_transformed = transformer.transform(df)
                self.logger.info(
                    f"Data transformed for {platform}: {len(df_transformed)} rows."
                    )
                transformed_data.append(df_transformed)
            except Exception as e:
                self.logger.error(
                    f"Error transforming data for {platform}: {e}"
                    )
                # Можно или пропустить, или завершить пайплайн с ошибкой
                return {
                    "status": "error",
                    "message": f"Transformation error: {platform}"
                }

        if not transformed_data:
            self.logger.warning(
                "No data after transformation. Nothing to merge."
                )
            return {
                "status": "no_data", "message": "No data after transformation."
            }

        # Шаг 3: Объединить данные
        unified_df = self.merger.merge(transformed_data)
        self.logger.info(f"Merged data shape: {unified_df.shape}")

        if unified_df.empty:
            self.logger.warning(
                "Unified DataFrame is empty after merge. Nothing to load."
                )
            return {
                "status": "no_data", "message": "Unified DataFrame is empty."
            }

        # Шаг 4: Финальные подготовка
        unified_df = transform_to_clickhouse_schema(unified_df)

        # Шаг 5: Загрузить данные
        try:
            loader = self.loaders.get(mode)
            self.logger.info(unified_df.info())
            loader.load(
                df=unified_df,
                table_name=self.clickhouse_config.table_name,
            )
            self.logger.info("Data successfully loaded")
            if mode == "production":
                return {
                    "status": "success",
                    "message": "Data loaded into ClickHouse."
                }
            else:
                return {
                    "status": "success",
                    "message": "CSV data loaded into output directory ."
                }
        except Exception as e:
            self.logger.error(f"Error loading data to ClickHouse: {e}")
            return {
                "status": "error", "message": f"ClickHouse load error: {e}"
            }

