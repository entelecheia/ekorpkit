import codecs
import logging
import pandas as pd
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class ColumnInfo:
    def __init__(self, **args):
        self.args = eKonf.to_dict(args)

        self._id_key = "id"
        self._text_key = "text"
        self._merge_meta_on_key = "merge_meta_on"
        self._timestamp_key = "timestamp"
        self._datetime_key = "datetime"
        self._split_key = "split"
        self._corpus_key = "corpus"
        self._dataset_key = "dataset"
        self._orginal_id_key = "_id"

        self.KEYs = self.KEYs
        self.COLUMNs = self.COLUMNs
        self.DATATYPEs = self.DATATYPEs
        self.METATYPEs = self.METATYPEs
        self.TIMESTAMP_PARM = self.TIMESTAMP_PARM
        self.DATETIME_PARM = self.DATETIME_PARM

        self.ID = self.ID
        self._ID = self._ID
        self.TEXT = self.TEXT
        self.TIMESTAMP = self.TIMESTAMP
        self.SPLIT = self.SPLIT
        self.CORPUS = self.CORPUS
        self.DATASET = self.DATASET

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.INFO}"
        return s

    def to_timestamp(self, data, metadata=None):
        if self.TIMESTAMP_PARM is None:
            return data, metadata

        _key = self.TIMESTAMP_PARM.get("key")
        _format = self.TIMESTAMP_PARM.get("format")
        _params = self.TIMESTAMP_PARM.get("params") or {}
        if _key is None:
            log.info("No timestamp key found")
            return data, metadata
        if isinstance(data, pd.DataFrame):
            if _key in data.columns:
                data[self.TIMESTAMP] = pd.to_datetime(
                    data[_key], format=_format, **_params
                )
                log.info(f"Loaded timestamp column {self.TIMESTAMP}")
            elif metadata is not None and _key in metadata.columns:
                metadata[self.TIMESTAMP] = pd.to_datetime(
                    metadata[_key], format=_format, **_params
                )
                df_dt = metadata[self.MERGE_META_ON + [self.TIMESTAMP]].copy()
                data = data.merge(df_dt, on=self.MERGE_META_ON, how="left")
                metadata.drop(self.TIMESTAMP, axis=1, inplace=True)
                log.info(f"Timestamp column {self.TIMESTAMP} added to data")
        return data, metadata

    def to_datetime(self, data):
        if self.DATETIME_PARM is None:
            return data

        _columns = _ensure_list(self.DATETIME_PARM.get("key"))
        _format = self.DATETIME_PARM.get("format", None)
        _params = self.DATETIME_PARM.get("params") or {}
        if _columns is None:
            log.info("No datetime column found")
            return data
        if isinstance(data, pd.DataFrame):
            for _col in _columns:
                if _col in data.columns:
                    data[_col] = pd.to_datetime(data[_col], format=_format, **_params)
                    log.info(f"converted datetime column {_col}")
        return data

    def combine_texts(self, data):
        if self.TEXTs is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.TEXTs] = data[self.TEXTs].fillna("")
            if len(self.TEXTs) > 1:
                data[self.TEXT] = data[self.TEXTs].apply(
                    lambda row: self.SEGMENT_SEP.join(row.values.astype(str)),
                    axis=1,
                )
                self.DATATYPEs = {
                    k: v for k, v in self.DATATYPEs.items() if k not in self.TEXTs
                }
                self.DATATYPEs[self.TEXT] = "str"

        return data

    def combine_ids(self, data):
        if self.IDs is None:
            return data

        if isinstance(data, pd.DataFrame):
            if len(self.IDS) > 1:
                data[self.ID] = data[self.IDs].apply(
                    lambda row: self.ID_SEPARATOR.join(row.values.astype(str)),
                    axis=1,
                )

        return data

    def merge_metadata(self, data, metadata):
        if metadata is None:
            return data
        data = data.merge(
            metadata,
            on=self.MERGE_META_ON,
            how="left",
        )
        return data

    def append_split(self, data, _split):
        if _split is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.SPLIT] = _split
            if self.SPLIT not in self.IDs:
                self.IDs.append(self.SPLIT)
            if self.DATA and self.SPLIT not in self.DATA:
                self.DATATYPEs[self.SPLIT] = "str"
            if self.METADATA and self.SPLIT not in self.METADATA:
                self.METATYPEs[self.SPLIT] = "str"

            log.info(f"Added split column [{self.SPLIT}] with value [{_split}]")

        return data

    def append_corpus(self, data, _corpus):
        if _corpus is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.CORPUS] = _corpus
            if self.CORPUS not in self.IDs:
                self.IDs.append(self.CORPUS)
            if self.DATA and self.CORPUS not in self.DATA:
                self.DATATYPEs[self.CORPUS] = "str"
            if self.METADATA and self.CORPUS not in self.METADATA:
                self.METATYPEs[self.CORPUS] = "str"

            log.info(f"Added corpus column [{self.CORPUS}] with value [{_corpus}]")

        return data

    def common_columns(self, dataframes):
        """
        Find common columns between dataframes
        """
        if not isinstance(dataframes, list):
            dataframes = [dataframes]
        common_columns = list(set.intersection(*(set(df.columns) for df in dataframes)))
        df = dataframes[0][common_columns].copy()
        dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
        self.DATATYPEs = dtypes
        return common_columns

    def append_dataset(self, data, _dataset):
        if _dataset is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.DATASET] = _dataset
            if self.DATASET not in self.IDs:
                self.IDs.append(self.DATASET)
            if self.DATA and self.DATASET not in self.DATA:
                self.DATATYPEs[self.DATASET] = "str"

            log.info(f"Added corpus column [{self.CORPUS}] with value [{_dataset}]")

        return data

    def reset_id(self, df):
        if isinstance(df, pd.DataFrame):
            df.rename({self.ID: self._ID}, inplace=True)
            df.reset_index().rename({"index": self.ID}, inplace=True)
            dtypes = df.dtypes.apply(lambda x: x.name).to_dict()
            self.DATATYPEs = dtypes
        return df

    @property
    def INFO(self):
        return self.args

    @property
    def ID(self):
        return self._id_key

    @ID.setter
    def ID(self, value):
        self.KEYs[self._id_key] = value

    @property
    def _ID(self):
        return self.KEYs.get(self._orginal_id_key) or self._orginal_id_key

    @_ID.setter
    def _ID(self, value):
        self.KEYs[self._orginal_id_key] = value

    @property
    def ID_SEPARATOR(self):
        return "_"

    @property
    def MERGE_META_ON(self):
        return _ensure_list(self.COLUMNs[self._merge_meta_on_key]) or self.IDs

    @MERGE_META_ON.setter
    def MERGE_META_ON(self, value):
        self.COLUMNs[self._merge_meta_on_key] = value

    @property
    def IDs(self):
        return _ensure_list(self.COLUMNs[self._id_key])

    @IDs.setter
    def IDs(self, value):
        self.COLUMNs[self._id_key] = value

    @property
    def TEXT(self):
        return self._text_key

    @TEXT.setter
    def TEXT(self, value):
        self.KEYs[self._text_key] = value

    @property
    def TEXTs(self):
        return _ensure_list(self.COLUMNs[self._text_key])

    @TEXTs.setter
    def TEXTs(self, value):
        self.COLUMNs[self._text_key] = value

    @property
    def DATA(self):
        if self.DATATYPEs is None:
            return None
        return list(self.DATATYPEs.keys())

    @property
    def METADATA(self):
        if self.METATYPEs is None:
            return None
        return list(self.METATYPEs.keys())

    @property
    def TIMESTAMP(self):
        return self._timestamp_key

    @TIMESTAMP.setter
    def TIMESTAMP(self, value):
        self.KEYs[self._timestamp_key] = value

    @property
    def SPLIT(self):
        return self.KEYs.get(self._split_key) or self._split_key

    @SPLIT.setter
    def SPLIT(self, value):
        self.KEYs[self._split_key] = value

    @property
    def CORPUS(self):
        return self.KEYs.get(self._corpus_key) or self._corpus_key

    @CORPUS.setter
    def CORPUS(self, value):
        self.KEYs[self._corpus_key] = value

    @property
    def DATASET(self):
        return self.KEYs.get(self._dataset_key) or self._dataset_key

    @DATASET.setter
    def DATASET(self, value):
        self.KEYs[self._dataset_key] = value

    @property
    def KEYs(self):
        return self.INFO.get("keys") or {}

    @KEYs.setter
    def KEYs(self, value):
        self.INFO["keys"] = value

    @property
    def COLUMNs(self):
        return self.INFO.get("columns") or {}

    @COLUMNs.setter
    def COLUMNs(self, value):
        self.INFO["columns"] = value

    @property
    def DATATYPEs(self):
        return self.INFO.get("data")

    @DATATYPEs.setter
    def DATATYPEs(self, value):
        self.INFO["data"] = value

    @property
    def METATYPEs(self):
        return self.INFO.get("meta")

    @METATYPEs.setter
    def METATYPEs(self, value):
        self.INFO["meta"] = value

    @property
    def TIMESTAMP_PARM(self):
        return self.INFO.get(self._timestamp_key)

    @TIMESTAMP_PARM.setter
    def TIMESTAMP_PARM(self, value):
        self.INFO[self._timestamp_key] = value

    @property
    def DATETIME_PARM(self):
        return self.INFO.get(self._datetime_key)

    @DATETIME_PARM.setter
    def DATETIME_PARM(self, value):
        self.INFO[self._datetime_key] = value

    @property
    def SEGMENT_SEP(self):
        return codecs.decode(
            self.INFO.get("segment_separator", "\n\n"), "unicode_escape"
        )


def _ensure_list(value):
    if not value:
        return []
    elif isinstance(value, str):
        return [value]
    return list(value)
