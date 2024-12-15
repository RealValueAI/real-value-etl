import uuid

import pandas as pd
import numpy as np


def transform_to_clickhouse_schema(df: pd.DataFrame) -> pd.DataFrame:
    df['listing_id'] = df['listing_id'].fillna(0).astype(np.uint64)
    df['listing_url'] = df['listing_url'].astype(str)
    df['price'] = df['price'].astype(np.float64)  # Для Decimal в ClickHouse
    df['price_per_sqm'] = df['price_per_sqm'].astype(np.float64)
    df['mortgage_rate'] = df['mortgage_rate'].fillna(0).astype(np.float32)
    df['address'] = df['address'].astype(str)
    df['address_id'] = df['address_id'].fillna(0).astype(np.uint64)
    df['area'] = df['area'].astype(np.float64)
    df['rooms'] = df['rooms'].fillna(0).astype(np.uint8)
    df['floor'] = df['floor'].fillna(0).astype(np.uint8)
    # df['description']
    # df['published_date']
    # df['updated_date']
    df['seller_id'] = df['seller_id'].fillna(0).astype(np.uint64)
    # df['seller_name_hash']
    # df['company_name']
    df['company_id'] = df['company_id'].fillna(0).astype(np.uint64)
    df['property_type'] = df['property_type'].astype('category')
    df['category'] = df['category'].astype('category')
    df['house_floors'] = df['house_floors'].fillna(0).astype(np.uint8)
    df['deal_type'] = df['deal_type'].fillna('Unknown').astype('category')
    df['discount_status'] = df['discount_status'].fillna('Unknown').astype(
        'category'
    )
    df['discount_value'] = df['discount_value'].fillna(0).astype(np.float64)
    df['placement_paid'] = df['placement_paid'].fillna(0).astype(np.uint8)
    df['big_card'] = df['big_card'].fillna(0).astype(np.uint8)
    df['pin_color'] = df['pin_color'].fillna(0).astype(np.uint8)
    df['longitude'] = df['longitude'].astype(np.float64)
    df['latitude'] = df['latitude'].astype(np.float64)
    df['subway_distances'] = df['subway_distances'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    df['subway_names'] = df['subway_names'].apply(
        lambda x: x if isinstance(x, list) else []
    )
    #df['photo_urls']
    df['monthly_payment'] = df['monthly_payment'].fillna(0).astype(np.float64)
    df['advance_payment'] = df['advance_payment'].fillna(0).astype(np.float64)
    df['auction_status'] = df['auction_status'].fillna(0).astype(np.float64)
    df['uid'] = df['uid'].fillna('').astype(str)
    df['platform_id'] = df['platform_id'].astype(np.uint8)
    # df['created_at']
    df['seller_type'] = df['seller_type'].fillna('UNKNOWN').astype(
        'category'
    )
    df['flat_type'] = df['flat_type'].fillna('UNKNOWN').astype(
        'category'
    )
    df['height'] = df['height'].fillna(0).astype(np.float64)
    df['area_rooms'] = df['area_rooms'].fillna(0).astype(np.float64)
    df['previous_price'] = df['previous_price'].fillna(0).astype(np.float64)
    df['renovation_offer'] = df['renovation_offer'].fillna('UNKNOWN').astype(str)
    # df['subway_time'] = df['subway_time']
    df['balcony_type'] = df['balcony_type'].fillna('UNKNOWN').astype(
        'category'
    )
    df['window_view'] = df['window_view'].fillna('UNKNOWN').astype('category')
    df['built_year_offer'] = df['built_year_offer'].fillna(0).astype(np.uint8)
    df['building_state'] = df['building_state'].fillna('UNKNOWN').astype(
        'category'
    )
    df['type_house_offer'] = df['type_house_offer'].fillna('UNKNOWN').astype(
        'category'
    )

    def generate_clickhouse_uuid(row):
        unique_string = f"{row['listing_id']}_{row['platform_id']}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, unique_string))
    df['uid'] = df.apply(generate_clickhouse_uuid, axis=1)

    df.drop(
        columns=[
            'subway_time',
    #         'big_card',
    #         'pin_color',
         ],
         inplace=True,
    )
    df.to_csv('merged.csv')
    return df
