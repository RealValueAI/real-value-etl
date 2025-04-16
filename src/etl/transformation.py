import json
import re
import time
import uuid
from typing import Optional

import numpy as np
import pandas as pd
import requests

from src.etl.extraction import BaseExtractor
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
        combined_df['valid'] = 0

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
        combined_df['subway_time'] = combined_df['subway_time'].apply(lambda x: json.dumps(x, ensure_ascii=False))
        # combined_df['balcony_type']
        # combined_df['window_view']
        # combined_df['built_year_offer']
        # combined_df['building_state']
        # combined_df['type_house_offer']
        combined_df['valid'] = 0

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


class AvitoTransformer(BaseTransformer):
    def __init__(self):
        self.logger = get_logger('avito_transformer_logger')
        self.logger.info("AvitoTransformer initialized.")

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_trans = df.copy()

        # Пропускаем дубликаты по URL (если есть)
        df_trans.drop_duplicates(subset=['url_offer'], keep='first', inplace=True)
        self.logger.info("Duplicates dropped.")

        # listing_id – берем из id_offer (приводим к целому)
        df_trans['Object ID'] = pd.to_numeric(df_trans['id_offer'], errors='coerce').astype('Int64')

        # listing_url – берем из url_offer
        df_trans['listing_url'] = df_trans['url_offer']

        # Цена – price_offer, приводим к числовому
        df_trans['Price'] = df_trans['price_offer']

        # Price per sqm – вычисляем как price / square_total_offer (если площадь > 0)
        df_trans['Price_per_sqm'] = df_trans.apply(
            lambda row: row['Price'] / pd.to_numeric(row[
                                                         'square_total_offer'], errors='coerce')
            if pd.notnull(row['square_total_offer']) and float(row['square_total_offer']) > 0 else np.nan,
            axis=1
        )

        df_trans['Mortgage Rate'] = np.nan
        df_trans['Address'] = df_trans['address_offer'].fillna('').astype(
            'string')
        df_trans['Address_id'] = df_trans['Address'].apply(lambda x: abs(
            hash(x)) % (10 ** 10))
        df_trans['Area'] = pd.to_numeric(df_trans['square_total_offer'],
                                         errors='coerce')
        df_trans['Rooms'] = pd.to_numeric(df_trans['rooms_offer'],
                                          errors='coerce')
        df_trans['Floor'] = pd.to_numeric(df_trans['floor_offer'],
                                          errors='coerce').astype('Int64')
        df_trans['Description'] = df_trans['description_offer'].fillna(
            '').astype('string')
        df_trans['Published Date'] = pd.to_datetime(df_trans['date_offer'], errors='coerce', utc=True)
        df_trans['Published Date'] = df_trans['Published Date'].dt.tz_localize(None)
        df_trans['Published Date'] = df_trans['Published Date'].fillna(pd.Timestamp('1970-01-01 00:00:00'))
        df_trans['Published Date'] = df_trans['Published Date'].dt.floor('s')
        df_trans['Updated Date'] = df_trans['Published Date']

        # Seller поля: используем seller
        df_trans['Seller ID'] = np.nan
        df_trans['Seller Name Hash'] = np.nan
        df_trans['Company Name'] = np.nan
        df_trans['Company ID'] = np.nan

        # Property Type – из type_offer, приводим к нижнему регистру (например, flat, etc.)
        df_trans['Property Type'] = df_trans['type_offer'].str.lower().fillna('unknown')

        # Category – задаём статически
        df_trans['Category'] = 'living'

        # House Floors – из floors_house
        df_trans['House Floors'] = pd.to_numeric(df_trans['floors_house'], errors='coerce').astype('Int64')

        # Deal Type – из sdelka_offer (например, sale, rent)
        df_trans['Deal Type'] = df_trans['sdelka_offer'].str.lower().fillna('sale')

        # Discount Status и Discount Value – отсутствуют, задаём NaN и 0
        df_trans['Discount Status'] = np.nan
        df_trans['Discount Value'] = 0

        # Placement Paid, Big Card, Pin Color – отсутствуют, задаём как 0 (false)
        df_trans['Placement Paid'] = 0
        df_trans['Big Card'] = 0
        df_trans['Pin Color'] = 0

        # Гео координаты – из latitude и longitude
        df_trans['Latitude'] = pd.to_numeric(df_trans['latitude'], errors='coerce')
        df_trans['Longitude'] = pd.to_numeric(df_trans['longitude'], errors='coerce')

        # Обработка метро:
        # subway_names – берем из metro_name1, metro_name2, metro_name3, отфильтровывая пустые значения
        df_trans['Subway Names'] = df_trans.apply(
            lambda row: [name for name in [row.get('metro_name1'), row.get('metro_name2'), row.get('metro_name3')]
                         if pd.notnull(name) and str(name).strip() != ''], axis=1
        )
        # subway_distances – берем из distance_to_metro1, distance_to_metro2, distance_to_metro3
        df_trans['Subway Distances'] = df_trans.apply(
            lambda row: [float(row.get('distance_to_metro1')) if pd.notnull(row.get('distance_to_metro1')) else np.nan,
                         float(row.get('distance_to_metro2')) if pd.notnull(row.get('distance_to_metro2')) else np.nan,
                         float(row.get('distance_to_metro3')) if pd.notnull(row.get('distance_to_metro3')) else np.nan],
            axis=1
        )
        # Можно отфильтровать NaN значения, если необходимо:
        df_trans['Subway Distances'] = df_trans['Subway Distances'].apply(
            lambda lst: [x for x in lst if not pd.isna(x)]
        )

        # Photos URLs – из photo_list_offer; используем _safe_eval для преобразования строки в список
        df_trans['Photos URLs'] = df_trans['photo_list_offer'].apply(self._safe_eval)

        df_trans['Monthly Payment'] = np.nan
        df_trans['Advance Payment'] = 0
        df_trans['Auction Status'] = np.nan
        df_trans['uid'] = np.nan

        # platform_id – задаём для Avito 2
        df_trans['platform_id'] = 2

        # created_at – текущая дата и время
        now = pd.Timestamp.now()
        df_trans['created_at'] = now

        # seller_type – из developer_offer, если имеется, иначе из seller; приводим к верхнему регистру
        df_trans['seller_type'] = df_trans['developer_offer'].fillna(df_trans['seller']).str.upper()

        # flat_type – из type_offer
        df_trans['flat_type'] = df_trans['type_offer'].str.lower()

        # height – из height_offer
        df_trans['height'] = pd.to_numeric(df_trans['height_offer'], errors='coerce')

        # area_rooms – из square_rooms_offer
        df_trans['area_rooms'] = pd.to_numeric(df_trans['square_rooms_offer'], errors='coerce')

        # previous_price – отсутствует, оставляем NaN
        df_trans['previous_price'] = np.nan

        # renovation_offer – из renovation_offer
        df_trans['renovation_offer'] = df_trans['renovation_offer'].fillna('')

        # balcony_type – из balcony_type
        df_trans['balcony_type'] = 'UNKNOWN'

        # window_view – из window_view
        df_trans['window_view'] = 'UNKNOWN'

        # built_year_offer – из built_year_offer
        df_trans['built_year_offer'] = pd.to_numeric(df_trans['built_year_offer'], errors='coerce').astype('Int64')

        # building_state – отсутствует
        df_trans['building_state'] = 'UNKNOWN'

        # type_house_offer – из type_house_offer
        df_trans['type_house_offer'] = df_trans['type_house_offer'].fillna('')
        df_trans['valid'] = 0
        # Удаляем строки, где критически важные поля отсутствуют
        df_trans.dropna(subset=['Price', 'Area', 'Rooms', 'Address'],
                        inplace=True)

        df_trans['subway_time'] = np.nan
        self.logger.info("Avito Transformation complete.")
        return df_trans

    def _safe_eval(self, value: str):
        try:
            if isinstance(value, str) and value.strip().startswith('['):
                return eval(value)
            return []
        except Exception as e:
            self.logger.error(f"Error during eval: {e}")
            return []