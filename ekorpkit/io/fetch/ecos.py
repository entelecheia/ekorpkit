from fileinput import filename
import logging
import logging
import os
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.fetch.base import BaseFetcher


log = logging.getLogger(__name__)


class ECOS(BaseFetcher):
    ecos_url = "http://ecos.bok.or.kr/api/StatisticSearch/"

    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        super().__init__(**args)

        self.ecos_api_key = self.args.get("ecos_api_key")
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


        if self.auto.load:
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

    def get_ecos(self, stat_cd, period, start_date, end_date):
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd
        from lxml import html
        from urllib.request import Request, urlopen
        from urllib.parse import urlencode, quote_plus, unquote

        url = self.ecos_url + "{}/xml/kr/1/30000/{}/{}/{}/{}/".format(
            self.ecos_api_key,
            stat_cd,
            period,
            start_date,
            end_date,
        )

        response = requests.get(url).content.decode("utf-8")

        xml_obj = BeautifulSoup(response, "lxml-xml")
        # xml_obj
        rows = xml_obj.findAll("row")
        return rows

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
        if not os.path.exists(filepath) or self.force.download:
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
