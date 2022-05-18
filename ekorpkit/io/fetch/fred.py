import logging
import logging
import os
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.file import save_dataframe, load_dataframe

log = logging.getLogger(__name__)


class Fred:
    def __init__(self, **args):
        from fredapi import Fred

        self.args = eKonf.to_dict(args)
        self.autoload = self.args.get("autoload", True)
        self.name = self.args.get("name")
        self.api_key = self.args.get("api_key")
        self.verbose = self.args.get("verbose", True)
        self.series_id = self.args.get("series_id")
        self.observation_start = self.args.get("observation_start")
        self.observation_end = self.args.get("observation_end")

        self.output_dir = self.args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = os.path.join(self.output_dir, self.args["output_file"])
        self.force_download = self.args["force_download"]

        self.fred = Fred(api_key=self.api_key)
        self.data = None

        if self.autoload:
            self.load()

    def get_series(
        self, series_id, observation_start=None, observation_end=None, **kwargs
    ):
        """
        Get data for a Fred series id. This fetches the latest known data, and is equivalent to get_series_latest_release()

        Parameters
        ----------
        series_id : str
            Fred series id such as 'CPIAUCSL'
        observation_start : datetime or datetime-like str such as '7/1/2014', optional
            earliest observation date
        observation_end : datetime or datetime-like str such as '7/1/2014', optional
            latest observation date
        kwargs : additional parameters
            Any additional parameters supported by FRED. You can see https://api.stlouisfed.org/docs/fred/series_observations.html for the full list

        Returns
        -------
        data : Series
            a Series where each index is the observation date and the value is the data for the Fred series
        """

    def load(self):
        if not os.path.exists(self.output_file) or self.force_download:
            self.data = self.load_fred()
        else:
            log.info(f"{self.output_file} already exists.")
            self.data = load_dataframe(self.output_file)
        return self.data

    def load_fred(self):
        series_name = self.name
        series_ids = self.series_id
        if isinstance(series_ids, str):
            series_ids = [series_ids]
        elif not isinstance(series_ids, list):
            series_ids = [None]

        _series = []
        for series_id in series_ids:
            series = self.fred.get_series(
                series_id=series_id,
                observation_start=self.observation_start,
                observation_end=self.observation_end,
            )
            _series.append(series)

        series = pd.concat(_series, axis=0)
        df = pd.DataFrame(series, columns=[series_name])
        save_dataframe(df, self.output_file)
        if self.verbose:
            print(df.tail())
        print(f"Saved {len(df.index)} records to {self.output_file}")
        return df
