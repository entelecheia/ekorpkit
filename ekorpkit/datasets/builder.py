from hyfi.config import BaseBatchModel


class DatasetBuilder(BaseBatchModel):
    name: str
    filetype: str = ".parquet"
    features: dict = {}

    class Config:
        underscore_attrs_are_private = False
