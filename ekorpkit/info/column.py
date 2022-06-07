import codecs
import logging
import pandas as pd
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class ColumnInfo:
    def __init__(self, **args):
        self.args = eKonf.to_dict(args)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.INFO}"
        return s

    def to_timestamp(self, data, metadata=None):
        if self.TIMESTAMP_PARM is None:
            return data, metadata

        _key = self.TIMESTAMP_PARM.get(eKonf.Keys.KEY)
        _format = self.TIMESTAMP_PARM.get(eKonf.Keys.FORMAT)
        _parms_ = self.TIMESTAMP_PARM.get(eKonf.Keys.PARMS) or {}
        if _key is None:
            log.info("No timestamp key found")
            return data, metadata
        if isinstance(data, pd.DataFrame):
            if _key in data.columns:
                data[self.TIMESTAMP] = pd.to_datetime(
                    data[_key], format=_format, **_parms_
                )
                log.info(f"Loaded timestamp column {self.TIMESTAMP}")
            elif metadata is not None and _key in metadata.columns:
                metadata[self.TIMESTAMP] = pd.to_datetime(
                    metadata[_key], format=_format, **_parms_
                )
                df_dt = metadata[self.MERGE_META_ON + [self.TIMESTAMP]].copy()
                data = data.merge(df_dt, on=self.MERGE_META_ON, how="left")
                metadata.drop(self.TIMESTAMP, axis=1, inplace=True)
                log.info(f"Timestamp column {self.TIMESTAMP} added to data")
        return data, metadata

    def to_datetime(self, data):
        if self.DATETIME_PARM is None:
            return data

        _columns = eKonf.ensure_list(self.DATETIME_PARM.get(eKonf.Keys.COLUMNS))
        _format = self.DATETIME_PARM.get(eKonf.Keys.FORMAT, None)
        _parms_ = self.DATETIME_PARM.get(eKonf.Keys.PARMS) or {}
        if _columns is None:
            log.info("No datetime column found")
            return data
        if isinstance(data, pd.DataFrame):
            for _col in _columns:
                if _col in data.columns:
                    data[_col] = pd.to_datetime(data[_col], format=_format, **_parms_)
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

            log.info(f"Added a column [{self.SPLIT}] with value [{_split}]")

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

            log.info(f"Added a column [{self.CORPUS}] with value [{_corpus}]")

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

            log.info(f"Added a column [{self.DATASET}] with value [{_dataset}]")

        return data

    def reset_id(self, data):
        if isinstance(data, pd.DataFrame):
            data.rename({self.ID: self._ID}, inplace=True)
            data.reset_index().rename({"index": self.ID}, inplace=True)
            dtypes = data.dtypes.apply(lambda x: x.name).to_dict()
            self.DATATYPEs = dtypes
        return data

    @property
    def INFO(self):
        return self.args

    @property
    def ID(self):
        return eKonf.Keys.ID.value

    @ID.setter
    def ID(self, value):
        self.KEYs[eKonf.Keys.ID] = value

    @property
    def _ID(self):
        return self.KEYs.get(eKonf.Keys._ID) or eKonf.Keys._ID.value

    @_ID.setter
    def _ID(self, value):
        self.KEYs[eKonf.Keys._ID] = value

    @property
    def ID_SEPARATOR(self):
        return "_"

    @property
    def MERGE_META_ON(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.META_MERGE_ON)) or self.IDs

    @MERGE_META_ON.setter
    def MERGE_META_ON(self, value):
        self.COLUMNs[eKonf.Keys.META_MERGE_ON] = value

    @property
    def IDs(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.ID))

    @IDs.setter
    def IDs(self, value):
        self.COLUMNs[eKonf.Keys.ID] = value

    @property
    def TEXT(self):
        return eKonf.Keys.TEXT.value

    @TEXT.setter
    def TEXT(self, value):
        self.KEYs[eKonf.Keys.TEXT] = value

    @property
    def TEXTs(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.TEXT))

    @TEXTs.setter
    def TEXTs(self, value):
        self.COLUMNs[eKonf.Keys.TEXT] = value

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
        return eKonf.Keys.TIMESTAMP.value

    @TIMESTAMP.setter
    def TIMESTAMP(self, value):
        self.KEYs[eKonf.Keys.TIMESTAMP] = value

    @property
    def SPLIT(self):
        return self.KEYs.get(eKonf.Keys.SPLIT) or eKonf.Keys.SPLIT.value

    @SPLIT.setter
    def SPLIT(self, value):
        self.KEYs[eKonf.Keys.SPLIT] = value

    @property
    def CORPUS(self):
        return self.KEYs.get(eKonf.Keys.CORPUS) or eKonf.Keys.CORPUS.value

    @CORPUS.setter
    def CORPUS(self, value):
        self.KEYs[eKonf.Keys.CORPUS] = value

    @property
    def DATASET(self):
        return self.KEYs.get(eKonf.Keys.DATASET) or eKonf.Keys.DATASET.value

    @DATASET.setter
    def DATASET(self, value):
        self.KEYs[eKonf.Keys.DATASET] = value

    @property
    def KEYs(self):
        return self.INFO.get(eKonf.Keys.KEYS) or {}

    @KEYs.setter
    def KEYs(self, value):
        self.INFO[eKonf.Keys.KEYS] = value

    @property
    def COLUMNs(self):
        return self.INFO.get(eKonf.Keys.COLUMNS) or {}

    @COLUMNs.setter
    def COLUMNs(self, value):
        self.INFO[eKonf.Keys.COLUMNS] = value

    @property
    def DATATYPEs(self):
        return self.INFO.get(eKonf.Keys.DATA)

    @DATATYPEs.setter
    def DATATYPEs(self, value):
        self.INFO[eKonf.Keys.DATA] = value

    @property
    def METATYPEs(self):
        return self.INFO.get(eKonf.Keys.META)

    @METATYPEs.setter
    def METATYPEs(self, value):
        self.INFO[eKonf.Keys.META] = value

    @property
    def TIMESTAMP_PARM(self):
        return self.INFO.get(eKonf.Keys.TIMESTAMP)

    @TIMESTAMP_PARM.setter
    def TIMESTAMP_PARM(self, value):
        self.INFO[eKonf.Keys.TIMESTAMP] = value

    @property
    def DATETIME_PARM(self):
        return self.INFO.get(eKonf.Keys.DATETIME)

    @DATETIME_PARM.setter
    def DATETIME_PARM(self, value):
        self.INFO[eKonf.Keys.DATETIME] = value

    @property
    def SEGMENT_SEP(self):
        return codecs.decode(
            self.INFO.get("segment_separator", "\n\n"), "unicode_escape"
        )

    @property
    def SENTENCE_SEP(self):
        return codecs.decode(
            self.INFO.get("sentence_separator", "\n"), "unicode_escape"
        )
