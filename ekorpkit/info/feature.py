import codecs
import logging
import pandas as pd
from ekorpkit import eKonf

log = logging.getLogger(__name__)


class FeatureInfo:
    def __init__(self, **args):
        self.args = eKonf.to_dict(args)

    def __str__(self):
        classname = self.__class__.__name__
        s = f"{classname} : {self.INFO}"
        return s

    def init_info(self, data):
        if isinstance(data, pd.DataFrame):
            if self.INDEX is None and data.index.name is None:
                self.INDEX = eKonf.Keys.INDEX.value
                data.index.name = self.INDEX
            elif self.INDEX is None:
                self.INDEX = data.index.name
            elif self.INDEX != data.index.name:
                data = self.set_index(data, self.INDEX)

            if not self.ID or self.ID[0] == eKonf.Keys.INDEX:
                self.ID = [self.INDEX]
            self.set_dtypes(data)
        return data

    def set_index(self, data, name):
        if isinstance(data, pd.DataFrame):
            if name in data.columns:
                data.set_index(name, inplace=True)
                self.INDEX = name
            else:
                log.warning(f"{name} not in dataframe")
        return data

    def reset_index(self, data):
        if isinstance(data, pd.DataFrame):
            if self.INDEX in data.columns:
                data.drop(self.INDEX, axis=1, inplace=True)
            data.reset_index(inplace=True)
            self.INDEX = eKonf.Keys.INDEX.value
            self.set_dtypes(data)
        return data

    def set_dtypes(self, data):
        if isinstance(data, pd.DataFrame):
            dtypes = data.dtypes.apply(lambda x: x.name).to_dict()
            self.DATATYPEs = dtypes
        return data

    def append_id(self, _id):
        log.info(f"Adding id [{_id}] to {self.ID}")
        if self.ID is None:
            self.ID = [_id]
        else:
            if isinstance(self.ID, str):
                self.ID = [self.ID]
            self.ID += [_id]
        log.info(f"Added id [{_id}], now {self.ID}")

    def append_split(self, data, _split):
        if _split is None:
            return data

        if isinstance(data, pd.DataFrame):
            data[self.SPLIT_KEY] = _split
            if self.SPLIT_KEY not in self.ID:
                self.append_id(self.SPLIT_KEY)
            if self.DATA and self.SPLIT_KEY not in self.DATA:
                self.DATATYPEs[self.SPLIT_KEY] = "str"

            log.info(f"Added a column [{self.SPLIT_KEY}] with value [{_split}]")

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
            if self.DATASET not in self.ID:
                self.append_id(self.DATASET)
            if self.DATA and self.DATASET not in self.DATA:
                self.DATATYPEs[self.DATASET] = "str"

            log.info(f"Added a column [{self.DATASET}] with value [{_dataset}]")

        return data

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

    @property
    def INFO(self):
        return self.args

    @property
    def ID_KEY(self):
        return eKonf.Keys.ID.value

    @ID_KEY.setter
    def ID_KEY(self, value):
        self.KEYs[eKonf.Keys.ID] = value

    @property
    def INDEX(self):
        return self.COLUMNs.get(eKonf.Keys.INDEX)

    @INDEX.setter
    def INDEX(self, value):
        self.COLUMNs[eKonf.Keys.INDEX] = value

    @property
    def ID(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.ID))

    @ID.setter
    def ID(self, value):
        self.COLUMNs[eKonf.Keys.ID] = value

    @property
    def Y(self):
        return self.COLUMNs.get(eKonf.Keys.Y)

    @Y.setter
    def Y(self, value):
        self.COLUMNs[eKonf.Keys.Y] = value

    @property
    def X(self):
        return eKonf.ensure_list(self.COLUMNs.get(eKonf.Keys.X))

    @X.setter
    def X(self, value):
        self.COLUMNs[eKonf.Keys.X] = value

    @property
    def SPLIT_KEY(self):
        return self.KEYs.get(eKonf.Keys.SPLIT) or eKonf.Keys.SPLIT.value

    @SPLIT_KEY.setter
    def SPLIT_KEY(self, value):
        self.KEYs[eKonf.Keys.SPLIT] = value

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
    def DATA(self):
        if self.DATATYPEs is None:
            return None
        return list(self.DATATYPEs.keys())

    @property
    def DATATYPEs(self):
        return self.INFO.get(eKonf.Keys.DATA)

    @DATATYPEs.setter
    def DATATYPEs(self, value):
        self.INFO[eKonf.Keys.DATA] = value

    @property
    def DATETIME_PARM(self):
        return self.INFO.get(eKonf.Keys.DATETIME)

    @DATETIME_PARM.setter
    def DATETIME_PARM(self, value):
        self.INFO[eKonf.Keys.DATETIME] = value
