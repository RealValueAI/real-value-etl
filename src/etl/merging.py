from typing import List
import pandas as pd
from src.utils.mapping import CH_FIELD_MAPPING


class UnifiedDataMerger:
    def __init__(self, field_mapping=CH_FIELD_MAPPING):
        self.field_mapping = field_mapping
        self.source_columns = list(self.field_mapping.keys())
        self.target_columns = list(self.field_mapping.values())

    def merge(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        if not dataframes:
            return pd.DataFrame(columns=self.target_columns)

        merged_df = pd.concat(dataframes, ignore_index=True, sort=False)

        for col in self.source_columns:
            if col not in merged_df.columns:
                merged_df[col] = pd.NA

        merged_df = merged_df[self.source_columns]
        merged_df = merged_df.rename(columns=self.field_mapping)

        # Располагаем столбцы в нужном порядке
        merged_df = merged_df[self.target_columns]

        return merged_df
