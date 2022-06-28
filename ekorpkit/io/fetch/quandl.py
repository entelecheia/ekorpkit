from fileinput import filename
import logging
import logging
import os
import pandas as pd
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class Quandl:
    def __init__(self, **args):
        from fredapi import Fred
        import nasdaqdatalink

        self.args = eKonf.to_dict(args)
        self.autoload = self.args.get("autoload", True)
        self.name = self.args.get("name")
        self.fred_api_key = self.args.get("fred_api_key")
        self.nasdaq_api_key = self.args.get("nasdaq_api_key")
        self.verbose = self.args.get("verbose", True)
        self.series_id = self.args.get("series_id")
        if isinstance(self.series_id, str):
            self.series_id = [self.series_id]
        elif not isinstance(self.series_id, list):
            self.series_id = []
        self.series_name = self.args.get("series_name")
        self.value_column = self.args.get("value_column") or "value"
        if self.series_name is None:
            if self.series_id:
                self.series_name = "_".join(self.series_id).replace("/", "_")
            else:
                self.series_name = self.value_column

        self.start_date = self.args.get("start_date")
        self.end_date = self.args.get("end_date")
        self.eval_columns = self.args.get("pipeline").get("eval_columns")

        self.output_dir = self.args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = self.args["output_file"]
        self.force_download = self.args["force_download"]

        self.fred = Fred(api_key=self.fred_api_key)
        nasdaqdatalink.ApiConfig.api_key = self.nasdaq_api_key
        self.data = None

        if self.autoload:
            self.load(
                series_id=self.series_id,
                series_name=self.series_name,
                start_date=self.start_date,
                end_date=self.end_date,
            )

    def get(self, series_id, start_date=None, end_date=None, **kwargs):
        if "/" in series_id:
            return self.get_nasqaq(series_id, start_date, end_date, **kwargs)
        else:
            series = self.get_series(series_id, start_date, end_date, **kwargs)
            df = pd.DataFrame(series, columns=[self.value_column])
            return df

    def get_nasqaq(self, series_id, start_date=None, end_date=None, **kwargs):
        """Return dataframe of requested dataset from Nasdaq Data Link.
        :param series_id: str or list, depending on single dataset usage or multiset usage
                Dataset codes are available on the Nasdaq Data Link website
        :param str api_key: Downloads are limited to 50 unless api_key is specified
        :param str start_date, end_date: Optional datefilers, otherwise entire
            dataset is returned
        :param str collapse: Options are daily, weekly, monthly, quarterly, annual
        :param str transform: options are diff, rdiff, cumul, and normalize
        :param int rows: Number of rows which will be returned
        :param str order: options are asc, desc. Default: `asc`
        :param str returns: specify what format you wish your dataset returned as,
            either `numpy` for a numpy ndarray or `pandas`. Default: `pandas`
        :returns: :class:`pandas.DataFrame` or :class:`numpy.ndarray`
        Note that Pandas expects timeseries data to be sorted ascending for most
        timeseries functionality to work.
        Any other `kwargs` passed to `get` are sent as field/value params to Nasdaq Data Link
        with no interference.
        """
        import nasdaqdatalink

        return nasdaqdatalink.get(
            dataset=series_id, start_date=start_date, end_date=end_date, **kwargs
        )

    def get_series(self, series_id, start_date=None, end_date=None, **kwargs):
        """
        Get data for a Fred series id. This fetches the latest known data, and is equivalent to get_series_latest_release()

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'
        start_date : datetime or datetime-like str such as '7/1/2014', optional
            earliest observation date
        end_date : datetime or datetime-like str such as '7/1/2014', optional
            latest observation date
        kwargs : additional parameters
            Any additional parameters supported by FRED. You can see https://api.stlouisfed.org/docs/fred/series_observations.html for the full list

        Returns
        -------
        data : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """
        return self.fred.get_series(
            series_id,
            observation_start=start_date,
            observation_end=end_date,
            **kwargs,
        )

    def load(
        self,
        series_id,
        series_name=None,
        start_date=None,
        end_date=None,
        filename=None,
        expressions=None,
        index_name="date",
        reset_index=False,
    ):
        if isinstance(series_id, str):
            series_id = [series_id]
        elif not isinstance(series_id, list):
            series_id = []

        if series_name is None:
            series_name = "_".join(series_id).replace("/", "_")
        if filename is None:
            filename = f"{series_name}.parquet"

        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date

        if isinstance(self.eval_columns, dict):
            self.eval_columns["expressions"] = expressions

        if self.verbose:
            print(f"Loading {series_name}{series_id} from {start_date} to {end_date}")

        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath) or self.force_download:
            self.data = self._load_series(
                series_id, series_name, start_date, end_date, index_name, reset_index
            )
            eKonf.save_data(self.data, filepath, verbose=self.verbose)
        else:
            log.info(f"{filepath} already exists.")
            self.data = eKonf.load_data(filepath, verbose=self.verbose)
        return self.data.copy()

    def _load_series(
        self,
        series_ids=None,
        series_name=None,
        start_date=None,
        end_date=None,
        index_name="date",
        reset_index=False,
        **kwargs,
    ):

        _dfs = []
        for series_id in series_ids:
            df = self.get(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                **kwargs,
            )
            if len(df.columns) == 1:
                if series_name is None:
                    series_name = df.columns[0]
                df.columns = [self.value_column]
            df = eKonf.pipe(df, self.eval_columns)
            if len(series_ids) > 1:
                df["series_id"] = series_id
            if series_name:
                columns = {
                    col: col.replace(self.value_column, series_name)
                    for col in df.columns
                    if col.startswith(self.value_column)
                }
                df.rename(columns=columns, inplace=True)
            if self.verbose:
                print(df.head())
            _dfs.append(df)

        df = pd.concat(_dfs)
        df.index.name = index_name
        if reset_index:
            df.index.name = series_name + "_" if series_name else "" + df.index.name
            df = df.reset_index()
        return df
