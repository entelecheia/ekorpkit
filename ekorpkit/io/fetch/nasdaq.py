from fileinput import filename
import logging
import logging
import os
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.file import save_dataframe, load_dataframe

log = logging.getLogger(__name__)


class NasdaqDataLink:
    def __init__(self, **args):
        import nasdaqdatalink

        self.args = eKonf.to_dict(args)
        self.autoload = self.args.get("autoload", True)
        self.name = self.args.get("name")
        self.api_key = self.args.get("api_key")
        self.verbose = self.args.get("verbose", True)
        self.series_id = self.args.get("series_id")
        if isinstance(self.series_id, str):
            self.series_id = [self.series_id]
        elif not isinstance(self.series_id, list):
            self.series_id = []
        self.series_name = self.args.get("series_name")
        self.value_column = self.args.get("value_column", "value")
        self.start_date = self.args.get("start_date")
        self.end_date = self.args.get("end_date")
        self.kwargs = self.args.get("kwargs") or {}
        self.eval_columns = self.args.get("pipeline").get("eval_columns")

        self.output_dir = self.args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = self.args["output_file"]
        self.force_download = self.args["force_download"]

        nasdaqdatalink.ApiConfig.api_key = self.api_key
        self.data = None

        if self.autoload:
            self.load()

    def get(self, series_id, start_date=None, end_date=None, **kwargs):
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

    def load(
        self,
        series_id=None,
        series_name=None,
        start_date=None,
        end_date=None,
        filename=None,
        **kwargs,
    ):
        if isinstance(series_id, str):
            series_id = [series_id]
        elif not isinstance(series_id, list):
            series_id = []

        if not series_id:
            series_id = self.series_id
        if series_name is None:
            if self.series_name:
                series_name = self.series_name
            else:
                series_name = "_".join(series_id)

        if filename is None:
            filename = f"{series_name}.parquet"

        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if not kwargs:
            kwargs = self.kwargs

        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath) or self.force_download:
            self.data = self._load_naqdaq(
                series_id, series_name, start_date, end_date, **kwargs
            )
            save_dataframe(self.data, filepath, verbose=self.verbose)
        else:
            log.info(f"{filepath} already exists.")
            self.data = load_dataframe(filepath, verbose=self.verbose)
        return self.data.copy()

    def _load_naqdaq(
        self,
        series_ids=None,
        series_name=None,
        start_date=None,
        end_date=None,
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
            df = eKonf.pipe(self.eval_columns, df)
            if len(series_ids) > 1:
                df["series_id"] = series_id
            if series_name:
                df.rename(columns={self.value_column: series_name}, inplace=True)
            _dfs.append(df)

        df = pd.concat(_dfs)
        return df
