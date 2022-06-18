import codecs
import logging
import pandas as pd
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class BaseInfo:

    Keys = eKonf.Keys

    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        self._initialized = False

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} :\n{self.INFO}"
        return s

    def init_info(self, data):
        if self._initialized:
            return data
        if isinstance(data, pd.DataFrame):
            log.info(
                f"index: {self.INDEX}, index of data: {data.index.name}, columns: {list(data.columns)}, id: {self.IDs}"
            )
            if data.index.name is None:
                data.index.name = self.INDEX
            elif self.INDEX is None:
                self.INDEX = data.index.name
            elif self.INDEX != data.index.name and self.INDEX in data.columns:
                data = self.set_index(data, self.INDEX)
            elif self.INDEX != data.index.name and self.INDEX not in data.columns:
                log.warning(f"{self.INDEX} not in dataframe")

            if not self.IDs or self.IDs[0] == self.Keys.INDEX.value:
                self.IDs = [self.INDEX]
            self.set_dtypes(data)
            self._initialized = True
        return data

    def set_index(self, data, name):
        if isinstance(data, pd.DataFrame):
            if name in data.columns:
                data.set_index(name, inplace=True)
                self.INDEX = name
            else:
                log.warning(f"{name} not in dataframe")
        return data

    def reset_index(
        self,
        data,
        rename_old_index=None,
        drop=False,
    ):
        if isinstance(data, pd.DataFrame):
            if self.INDEX in data.columns:
                data.drop(self.INDEX, axis=1, inplace=True)
            data = data.reset_index(drop=drop)
            if not drop and rename_old_index is not None and self.INDEX in data.columns:
                data = data.rename(columns={self.INDEX: rename_old_index})
            self.INDEX = self.Keys.INDEX.value
            self.set_dtypes(data)
        return data

    def reset_id(self, data):
        if isinstance(data, pd.DataFrame):
            data.rename(columns={self.ID: self._ID}, inplace=True)
            data = self.reset_index(data, rename_old_index=self.ID)
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

    def to_datetime(self, data):
        if self.DATETIME_INFO is None:
            return data

        _columns = eKonf.ensure_list(self.DATETIME_INFO.get(eKonf.Keys.COLUMNS))
        _format = self.DATETIME_INFO.get(eKonf.Keys.FORMAT, None)
        rcParams = self.DATETIME_INFO.get(eKonf.Keys.rcPARAMS) or {}
        if _columns is None:
            log.info("No datetime column found")
            return data
        if isinstance(data, pd.DataFrame):
            for _col in _columns:
                if _col in data.columns:
                    data[_col] = pd.to_datetime(data[_col], format=_format, **rcParams)
                    log.info(f"converted datetime column {_col}")
        return data

    def append_id(self, _id):
        log.info(f"Adding id [{_id}] to {self.IDs}")
        if self.IDs is None:
            self.IDs = [_id]
        else:
            if isinstance(self.IDs, str):
                self.IDs = [self.IDs]
            self.IDs += [_id]
        log.info(f"Added id [{_id}], now {self.IDs}")

    def append_dataset(self, data, _dataset):
        if _dataset is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.DATASET] = _dataset
            if self.DATASET not in self.IDs:
                self.append_id(self.DATASET)
            if self.DATA and self.DATASET not in self.DATA:
                self.DATATYPEs[self.DATASET] = "str"

            log.info(f"Added a column [{self.DATASET}] with value [{_dataset}]")

        return data

    def append_split(self, data, _split):
        if _split is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.SPLIT] = _split
            if self.SPLIT not in self.IDs:
                self.append_id(self.SPLIT)
            if self.DATA and self.SPLIT not in self.DATA:
                self.DATATYPEs[self.SPLIT] = "str"

            log.info(f"Added a column [{self.SPLIT}] with value [{_split}]")

        return data

    def set_dtypes(self, data):
        if isinstance(data, pd.DataFrame):
            dtypes = data.dtypes.apply(lambda x: x.name).to_dict()
            self.DATATYPEs = dtypes
        return data

    @property
    def _ID(self):
        return self.Keys._ID.value

    @property
    def ID_SEPARATOR(self):
        return eKonf.Defaults.ID_SEP.value

    @property
    def INFO(self):
        return self.args

    @property
    def DATETIME_INFO(self):
        return self.INFO.get(self.Keys.DATETIME)

    @DATETIME_INFO.setter
    def DATETIME_INFO(self, value):
        self.INFO[eKonf.Keys.DATETIME.value] = value

    @property
    def DATATYPEs(self):
        return self.INFO.get(eKonf.Keys.DATA)

    @DATATYPEs.setter
    def DATATYPEs(self, value):
        self.INFO[eKonf.Keys.DATA.value] = value

    @property
    def COLUMNs(self):
        return self.INFO.get(eKonf.Keys.COLUMNS) or {}

    @COLUMNs.setter
    def COLUMNs(self, value):
        self.INFO[eKonf.Keys.COLUMNS.value] = value

    @property
    def DATA(self):
        if self.DATATYPEs is None:
            return None
        return list(self.DATATYPEs.keys())

    @property
    def DATASET(self):
        return eKonf.Keys.DATASET.value

    @property
    def INDEX(self):
        return self.COLUMNs.get(eKonf.Keys.INDEX) or eKonf.Keys.INDEX.value

    @INDEX.setter
    def INDEX(self, value):
        self.COLUMNs[eKonf.Keys.INDEX.value] = value

    @property
    def ID(self):
        return eKonf.Keys.ID.value

    @property
    def IDs(self):
        return eKonf.ensure_list(self.COLUMNs.get(self.ID))

    @IDs.setter
    def IDs(self, value):
        self.COLUMNs[self.ID] = value

    @property
    def SPLIT(self):
        return eKonf.Keys.SPLIT.value


class CorpusInfo(BaseInfo):
    def __init__(self, **args):
        super().__init__(**args)

    def to_timestamp(self, data, metadata=None):
        if self.TIMESTAMP_INFO is None:
            return data, metadata

        _key = self.TIMESTAMP_INFO.get(eKonf.Keys.KEY)
        _format = self.TIMESTAMP_INFO.get(eKonf.Keys.FORMAT)
        rcParams = self.TIMESTAMP_INFO.get(eKonf.Keys.rcPARAMS) or {}
        if _key is None:
            log.info("No timestamp key found")
            return data, metadata
        if isinstance(data, pd.DataFrame):
            if _key in data.columns:
                data[self.TIMESTAMP] = pd.to_datetime(
                    data[_key], format=_format, **rcParams
                )
                log.info(f"Loaded timestamp column {self.TIMESTAMP}")
            elif metadata is not None and _key in metadata.columns:
                metadata[self.TIMESTAMP] = pd.to_datetime(
                    metadata[_key], format=_format, **rcParams
                )
                df_dt = metadata[self.MERGE_META_ON + [self.TIMESTAMP]].copy()
                data = data.merge(df_dt, on=self.MERGE_META_ON, how="left")
                # metadata.drop(self.TIMESTAMP, axis=1, inplace=True)
                log.info(f"Timestamp column {self.TIMESTAMP} added to data")
        return data, metadata

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

    def merge_metadata(self, data, metadata):
        if metadata is None:
            return data
        meta_cols = [col for col in metadata.columns if col not in data.columns]
        meta_cols += self.MERGE_META_ON
        data = data.merge(metadata[meta_cols], on=self.MERGE_META_ON, how="left")
        return data

    def append_split_to_meta(self, metadata, _split):
        if _split is None:
            return metadata

        if isinstance(metadata, pd.DataFrame):
            metadata[self.SPLIT] = _split
            if self.METADATA and self.SPLIT not in self.METADATA:
                self.METATYPEs[self.SPLIT] = "str"

            log.info(f"Added a column [{self.SPLIT}] with value [{_split}]")

        return metadata

    def append_corpus(self, data, _corpus):
        if _corpus is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.CORPUS] = _corpus
            if self.CORPUS not in self.IDs:
                self.append_id(self.CORPUS)
            if self.DATA and self.CORPUS not in self.DATA:
                self.DATATYPEs[self.CORPUS] = "str"
            if self.METADATA and self.CORPUS not in self.METADATA:
                self.METATYPEs[self.CORPUS] = "str"

            log.info(f"Added a column [{self.CORPUS}] with value [{_corpus}]")

        return data

    @property
    def MERGE_META_ON(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.META_MERGE_ON)) or self.IDs

    @MERGE_META_ON.setter
    def MERGE_META_ON(self, value):
        self.COLUMNs[eKonf.Keys.META_MERGE_ON.value] = value

    @property
    def TEXT(self):
        return eKonf.Keys.TEXT.value

    @property
    def TEXTs(self):
        return eKonf.ensure_list(self.COLUMNs.get(self.TEXT))

    @TEXTs.setter
    def TEXTs(self, value):
        self.COLUMNs[self.TEXT] = value

    @property
    def METADATA(self):
        if self.METATYPEs is None:
            return None
        return list(self.METATYPEs.keys())

    @property
    def TIMESTAMP(self):
        return eKonf.Keys.TIMESTAMP.value

    @property
    def CORPUS(self):
        return eKonf.Keys.CORPUS.value

    @property
    def METATYPEs(self):
        return self.INFO.get(eKonf.Keys.META)

    @METATYPEs.setter
    def METATYPEs(self, value):
        self.INFO[eKonf.Keys.META.value] = value

    @property
    def TIMESTAMP_INFO(self):
        return self.INFO.get(self.TIMESTAMP)

    @TIMESTAMP_INFO.setter
    def TIMESTAMP_INFO(self, value):
        self.INFO[self.TIMESTAMP] = value

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


class DatasetInfo(BaseInfo):
    def __init__(self, **args):
        super().__init__(**args)


class FeatureInfo(BaseInfo):
    def __init__(self, **args):
        super().__init__(**args)

    @property
    def Y(self):
        return self.COLUMNs.get(eKonf.Keys.Y)

    @Y.setter
    def Y(self, value):
        self.COLUMNs[eKonf.Keys.Y.value] = value

    @property
    def X(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.X))

    @X.setter
    def X(self, value):
        self.COLUMNs[eKonf.Keys.X.value] = value
