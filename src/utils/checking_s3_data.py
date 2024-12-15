import re
from datetime import datetime
from logging import Logger

import boto3
from typing import List, Optional, Dict

from src.utils.config import S3Config
from src.utils.logger import get_logger


class PlatformsDateResolver:
    def __init__(
        self,
        platforms: List[str],
        s3_config: S3Config,
    ):
        self.logger = get_logger(self.__class__.__name__.lower())
        self.s3_config = s3_config
        if (not self.s3_config.access_key or
                not self.s3_config.secret_key or
                not self.s3_config.bucket_name):
            self.logger.error(
                "S3 configuration is incomplete."
                "Check access_key, secret_key, and bucket_name."
                )
            raise ValueError("Invalid S3 configuration")
        self.platforms = platforms
        self.s3 = boto3.client(
            's3',
            aws_access_key_id=self.s3_config.access_key,
            aws_secret_access_key=self.s3_config.secret_key
        )
        self._check_s3_access()
        self.latest_dates = self._get_latest_dates()
        self.logger.info(f"Latest dates from S3: {self.latest_dates}")

    def _check_s3_access(self):
        """
        Проверяем, есть ли доступ к бакету, пытаясь получить его метаданные.
        Если нет доступа, выбрасываем исключение.
        """
        try:
            self.s3.head_bucket(Bucket=self.s3_config.bucket_name)
            self.logger.info(
                f"Access to S3 bucket '{self.s3_config.bucket_name}' verified."
                )
        except self.s3.exceptions.ClientError as e:
            self.logger.error(
                f"Cannot access S3 bucket '{self.s3_config.bucket_name}': {e}"
                )
            raise ValueError(
                f"Failed to access S3 bucket '{self.s3_config.bucket_name}'."
                f" Check credentials and bucket name."
                )

    def _get_latest_dates(self) -> Dict[str, Optional[datetime.date]]:
        response = self.s3.list_objects_v2(
            Bucket=self.s3_config.bucket_name,
            Prefix=self.s3_config.folder_name,
        )
        self.logger.info(response)
        platform_dates = {}

        # Регулярка для парсинга имени файла: <platform>_<YYYYMMDD>.csv
        f_name = self.s3_config.folder_name
        pattern = re.compile(
            fr'^{f_name}(?P<platform>\w+)_(?P<date>\d{{8}})\.csv$'
            )

        for obj in response.get('Contents', []):
            filename = obj['Key']  # например: domclick_20241117.csv
            match = pattern.match(filename)
            if match:
                platform = match.group('platform')
                date_str = match.group('date')
                file_date = datetime.strptime(date_str, '%Y%m%d').date()

                if platform not in platform_dates:
                    platform_dates[platform] = []
                platform_dates[platform].append(file_date)

        # Последняя дата для каждой платформы из списка известных
        latest_dates = {}
        for p in self.platforms:
            if p in platform_dates:
                latest_dates[p] = max(platform_dates[p])
            else:
                # Нет ни одного файла для этой платформы
                latest_dates[p] = None

        return latest_dates

    def resolve_dates(
        self,
        request_body: Dict[str, Optional[str]]
    ) -> Dict[str, Optional[str]]:
        """
        Логика выбора дат для каждой платформы:
        - Если в запросе указано "latest":
            берем последнюю доступную дату из self.latest_dates[p] (если None, значит данных нет)
        - Если указано "skip" или None:
            эту платформу не добавляем в результат (пропускаем)
        - Если указана конкретная дата (формат YYYYMMDD):
            проверяем, есть ли такая дата среди файлов (для упрощения будем считать,
            что если указанная дата равна последней доступной в latest_dates или раньше, значит используем её,
            иначе пропускаем.
        """

        result = {}
        for platform in self.platforms:
            user_value = request_body.get(
                platform, None
            )  # значение из запроса
            latest_date = self.latest_dates.get(platform, None)

            if user_value is None or user_value == "skip":
                continue
            elif user_value == "latest":
                if latest_date is not None:
                    result[platform] = latest_date.strftime("%Y%m%d")
                else:
                    result[platform] = None
            else:
                try:
                    requested_date = datetime.strptime(
                        user_value, "%Y%m%d"
                    ).date()
                except ValueError:
                    # Неверный формат даты, пропускаем платформу
                    self.logger.warning(
                        f"Invalid date format for platform {platform}: {user_value}"
                    )
                    continue

                # Проверяем, можем ли мы использовать эту дату
                # Логика: если latest_date не None и requested_date <= latest_date, используем её
                if latest_date is not None and requested_date <= latest_date:
                    result[platform] = requested_date.strftime("%Y%m%d")
                else:
                    continue

        self.logger.info(f"Data selected by resolver: {result}")
        return result
