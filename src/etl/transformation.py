import re
import time
import uuid

import numpy as np
import pandas as pd
import requests
from src.utils.logger import get_logger


class BaseTransformer:
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this!")


class DomClickTransformer(BaseTransformer):
    BASE_IMAGE_URL = "https://img.dmclk.ru/"
    BASE_SALE_URL = "https://domclick.ru/card/sale__flat__"

    def __init__(self):
        self.logger = get_logger('domclick_transformer_logger')
        self.logger.info("DomClickTransformer initialized.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Объединение всех DataFrame в один (если нужно)
        combined_df = df.copy()

        # Приведение типов
        combined_df['Object ID'] = np.floor(
            pd.to_numeric(combined_df['Object ID'], errors='coerce')
        ).astype('Int64')
        combined_df['listing_url'] = combined_df['Object ID'].apply(
            lambda x: f"{self.BASE_SALE_URL}{x}"
        )
        combined_df['Price'] = pd.to_numeric(
            combined_df['Price'], errors='coerce'
        )
        combined_df['Price per sqm'] = pd.to_numeric(
            combined_df['Price per sqm'], errors='coerce'
        )
        combined_df['Mortgage Rate'] = pd.to_numeric(
            combined_df['Mortgage Rate'], errors='coerce'
        )
        combined_df['Address'] = combined_df['Address'].fillna('').astype(
            'string'
        )
        combined_df['Address ID'] = pd.to_numeric(
            combined_df['Address ID'], errors='coerce'
        ).astype('Int64')
        combined_df['Area'] = pd.to_numeric(
            combined_df['Area'], errors='coerce'
        )
        combined_df['Rooms'] = pd.to_numeric(
            combined_df['Rooms'], errors='coerce'
        )
        combined_df['Floor'] = np.floor(
            pd.to_numeric(combined_df['Floor'], errors='coerce')
        ).astype('Int64')
        combined_df['Description'] = combined_df['Description'].fillna(
            ''
        ).astype('string')

        # Преобразование Published Date
        combined_df['Published Date'] = pd.to_datetime(
            combined_df['Published Date'], errors='coerce', utc=True
        )
        combined_df['Published Date'] = combined_df[
            'Published Date'].dt.tz_localize(None)  # Убираем временную зону
        combined_df['Published Date'] = combined_df['Published Date'].fillna(
            pd.Timestamp('1970-01-01 00:00:00')
        )
        combined_df['Published Date'] = combined_df['Published Date'].dt.floor(
            's'
        )
        # Преобразование Updated Date
        combined_df['Updated Date'] = pd.to_datetime(
            combined_df['Updated Date'], errors='coerce', utc=True
        )
        combined_df['Updated Date'] = combined_df[
            'Updated Date'].dt.tz_localize(None)  # Убираем временную зону
        combined_df['Updated Date'] = combined_df['Updated Date'].fillna(
            pd.Timestamp('1970-01-01 00:00:00')
        )
        combined_df['Updated Date'] = combined_df['Updated Date'].dt.floor('s')

        combined_df['Seller ID'] = pd.to_numeric(
            combined_df['Seller ID'], errors='coerce'
        ).astype('Int64')
        combined_df['Seller Name Hash'] = combined_df[
            'Seller Name Hash'].fillna('').astype('string')
        combined_df['Company Name'] = combined_df['Company Name'].fillna(
            ''
        ).astype('string')
        combined_df['Company ID'] = pd.to_numeric(
            combined_df['Company ID'], errors='coerce'
        ).fillna(
            combined_df['Company Name'].apply(
                lambda x: abs(hash(x)) % (10 ** 10)
            )
        ).astype('Int64')
        combined_df['Property Type'] = combined_df['Property Type'].fillna(
            'Unknown'
        )
        combined_df['Category'] = combined_df['Category'].fillna('Unknown')
        combined_df['House Floors'] = pd.to_numeric(
            combined_df['House Floors'], errors='coerce'
        ).astype('Int64')
        combined_df['Deal Type'] = combined_df['Deal Type'].fillna('Unknown')
        combined_df['Discount Status'] = combined_df['Discount Status'].fillna(
            'Unknown'
        )
        combined_df['Discount Value'] = pd.to_numeric(
            combined_df['Discount Value'], errors='coerce'
        ).fillna(0)
        combined_df['Placement Paid'] = combined_df['Placement Paid'].apply(
            lambda x: 1 if pd.notna(x) and x else 0
        ).astype('Float64')
        combined_df['Big Card'] = combined_df['Big Card'].apply(
            lambda x: 1 if pd.notna(x) and x else 0
        ).astype('Float64')
        combined_df['Pin Color'] = combined_df['Pin Color']
        combined_df['Longitude'] = pd.to_numeric(
            combined_df['Longitude'], errors='coerce'
        )
        combined_df['Latitude'] = pd.to_numeric(
            combined_df['Latitude'], errors='coerce'
        )
        combined_df['Subway Distances'] = combined_df[
            'Subway Distances'].apply(
            lambda x: self._safe_eval(x)
        )
        combined_df['Subway Names'] = combined_df['Subway Names'].apply(
            lambda x: self._safe_eval(x)
        )
        combined_df['Photos URLs'] = combined_df['Photos URLs'].apply(
            lambda x: self._safe_eval(x)
        )
        combined_df['Monthly Payment'] = pd.to_numeric(
            combined_df['Monthly Payment'], errors='coerce'
        ).fillna(0)
        combined_df['Advance Payment'] = pd.to_numeric(
            combined_df['Advance Payment'], errors='coerce'
        ).fillna(0)
        combined_df['Auction Status'] = combined_df['Auction Status']

        # Удаление записей с отсутствующими важными значениями
        combined_df.dropna(
            subset=['Object ID', 'Price', 'Area', 'Rooms', 'Address'],
            inplace=True
        )

        # Обработка URL фото
        combined_df['Photos URLs'] = combined_df['Photos URLs'].apply(
            self._process_photo_urls
        )
        self.logger.info("Processed photo URLs.")

        # Добавление дополнительных полей
        combined_df['uid'] = np.nan
        combined_df['platform_id'] = 1  # Для DomClick platform_id = 1
        combined_df['created_at'] = pd.Timestamp.now()

        # Новые колонки от яндекса
        combined_df['seller_type'] = np.nan
        combined_df['flat_type'] = np.nan
        combined_df['height'] = np.nan
        combined_df['area_rooms'] = np.nan
        combined_df['previous_price'] = np.nan
        combined_df['renovation_offer'] = np.nan
        combined_df['subway_time'] = np.nan
        combined_df['balcony_type'] = np.nan
        combined_df['window_view'] = np.nan
        combined_df['built_year_offer'] = np.nan
        combined_df['building_state'] = np.nan
        combined_df['type_house_offer'] = np.nan

        self.logger.info("DomClick Transformation complete.")
        return combined_df

    def _process_photo_urls(self, urls: list) -> list:
        # Добавление базового URL к каждой ссылке на фото
        return [self.BASE_IMAGE_URL + url for url in urls]

    def _validate_photo_urls(self, urls: list) -> list:
        # Проверка доступности каждого фото по URL
        valid_urls = []
        max_photos = min(len(urls), 3)  # Ограничение на количество
        for url in urls[:max_photos]:
            try:
                response = requests.head(url)
                if response.status_code == 200:
                    valid_urls.append(url)
                else:
                    self.logger.warning(f"Photo not accessible: {url}")
            except requests.RequestException as e:
                self.logger.error(f"Error while checking photo URL {url}: {e}")
        return valid_urls

    def _safe_eval(self, value: str):
        # Безопасное выполнение eval для строк, представляющих списки
        try:
            if isinstance(value, str) and value.startswith('['):
                return eval(value)
            return []
        except Exception as e:
            self.logger.error(f"Error during eval: {e}")
            return []


class YandexTransformer(BaseTransformer):
    BASE_URL = "https:"

    def __init__(self):
        self.logger = get_logger('yandex_transformer_logger')
        self.logger.info("YandexTransformer initialized.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        combined_df = df.copy()
        combined_df.drop_duplicates(
            subset=['url_offer_yand'],
            keep='first',
            inplace=True
        )
        self.logger.info('copy')
        # Приведение типов и вычисление значений
        combined_df['Object ID'] = combined_df['url_offer_yand'].apply(
            self._extract_object_id
        ).astype('Int64')
        self.logger.info('add object id')
        combined_df['listing_url'] = combined_df['url_offer_yand'].apply(
            lambda x: f"{self.BASE_URL}{x}"
        )
        self.logger.info('add listing url')
        combined_df['Price'] = pd.to_numeric(
            combined_df['price_offer'], errors='coerce'
        )
        combined_df['Price per sqm'] = pd.to_numeric(
            combined_df['price_offer'], errors='coerce'
        ) / pd.to_numeric(
            combined_df['square_total_offer'], errors='coerce'
        )
        self.logger.info('add prices')
        combined_df['Mortgage Rate'] = np.nan
        combined_df['Address'] = combined_df['address_offer'].fillna(
            ''
        ).astype('string')
        self.logger.info('add address')
        combined_df['Address ID'] = np.nan
        self.logger.info('add address id')
        combined_df['Area'] = pd.to_numeric(
            combined_df['square_total_offer'],
            errors='coerce'
        )
        self.logger.info('add area')
        combined_df['Rooms'] = pd.to_numeric(
            combined_df['rooms_offer'], errors='coerce'
        )
        self.logger.info('add rooms')
        combined_df['Floor'] = pd.to_numeric(
            combined_df['floor_offer'], errors='coerce'
        ).astype('Int64')
        self.logger.info('add floor')
        combined_df['Description'] = combined_df['description_offer'].fillna(
            ''
        ).astype('string')
        self.logger.info('add description')
        combined_df['Published Date'] = pd.to_datetime(
            combined_df['date_offer'], errors='coerce', utc=True
        )
        combined_df['Published Date'] = combined_df[
            'Published Date'].dt.tz_localize(None)  # Убираем временную зону
        combined_df['Published Date'] = combined_df['Published Date'].fillna(
            pd.Timestamp('1970-01-01 00:00:00')
        )
        combined_df['Published Date'] = combined_df['Published Date'].dt.floor(
            's'
        )
        self.logger.info('add date')
        combined_df['Updated Date'] = combined_df['Published Date']

        combined_df['Seller ID'] = np.nan
        combined_df['Seller Name Hash'] = np.nan
        combined_df['Company Name'] = np.nan
        combined_df['Company ID'] = np.nan
        combined_df['Property Type'] = np.where(
            combined_df['type_offer'] == 'NEW_FLAT', 'layout', 'flat'
        )
        combined_df['Category'] = 'living'
        combined_df['House Floors'] = pd.to_numeric(
            combined_df['floors_house'], errors='coerce'
        ).astype('Int64')
        combined_df['Deal Type'] = 'sale'
        combined_df['Discount Status'] = np.nan
        combined_df['Discount Value'] = np.nan
        combined_df['Placement Paid'] = np.nan
        combined_df['Big Card'] = np.nan
        combined_df['Pin Color'] = np.nan
        self.logger.info('add NANs')
        combined_df['Longitude'] = pd.to_numeric(
            combined_df['longitude'], errors='coerce'
        )
        combined_df['Latitude'] = pd.to_numeric(
            combined_df['latitude'], errors='coerce'
        )
        self.logger.info('add geo')
        combined_df['Subway Distances'] = np.nan
        self.logger.info('add distance')
        combined_df['Subway Names'] = combined_df['metro_name'].apply(
            lambda x: [x]
            )
        self.logger.info('add time')
        combined_df['Photos URLs'] = combined_df['photo_list_offer'].apply(
            lambda x: self._process_photo_urls(self._safe_eval(x))
        )
        combined_df['Monthly Payment'] = np.nan
        combined_df['Advance Payment'] = np.nan
        combined_df['Auction Status'] = np.nan
        combined_df['uid'] = np.nan
        combined_df['platform_id'] = 4  # Для Yandex платформы platform_id = 2
        combined_df['created_at'] = pd.Timestamp.now()
        self.logger.info('add uid')

        combined_df.dropna(
            subset=['Price', 'Area', 'Rooms', 'Address'],
            inplace=True
        )

        # Новые колонки от яндекса
        combined_df['seller_type'] = combined_df['seller']
        combined_df['flat_type'] = combined_df['type_offer']
        combined_df['height'] = combined_df['height_offer']
        combined_df['area_rooms'] = combined_df['square_rooms_offer']
        combined_df['previous_price'] = combined_df['previous_price_offer']
        #combined_df['renovation_offer']
        combined_df['subway_time'] = combined_df.apply(
            lambda row: {
                row['metro_name']: [row['metro_transp'], row['time_to_metro']]
            },
            axis=1
        )
        # combined_df['balcony_type']
        # combined_df['window_view']
        # combined_df['built_year_offer']
        # combined_df['building_state']
        # combined_df['type_house_offer']


        self.logger.info("Yandex Transformation complete.")
        return combined_df

    def _process_photo_urls(self, urls: list) -> list:
        # Добавление базового URL к каждой ссылке на фото
        return [self.BASE_URL + url.lstrip('/') for url in urls]

    def _safe_eval(self, value: str):
        # Безопасное выполнение eval для строк, представляющих списки
        try:
            if isinstance(value, str) and value.startswith('['):
                return eval(value)
            return []
        except Exception as e:
            self.logger.error(f"Error during eval: {e}")
            return []

    def _extract_object_id(self, url: str) -> int:
        """
        Извлекает Object ID из строки URL.
        Пример:
        //realty.yandex.ru/offer/5227641546799531676 -> 5227641546799531676
        """
        try:
            match = re.search(r'/offer/(\d+)', url)
            return int(match.group(1)) if match else None
        except Exception as e:
            self.logger.error(
                f"Error extracting Object ID from URL {url}: {e}"
            )
