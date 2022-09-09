import logging
import os
import json
import pandas as pd
from ekorpkit import eKonf
from ekorpkit.io.fetch.base import BaseFetcher


log = logging.getLogger(__name__)


class ECOS(BaseFetcher):
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        super().__init__(**args)

        self._ecos_api_key = self.args.get("ecos_api_key")
        self._api = self.args.get("api")

        self.series_id = self.args.get("series_id")
        self.value_column = self.args.get("value_column") or "value"
        self.start_date = self.args.get("start_date")
        self.end_date = self.args.get("end_date")
        self.cycle = self.args.get("cycle")
        self.eval_columns = self.args.get("pipeline").get("eval_columns")

        self.output_dir = self.args["output_dir"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_file = self.args["output_file"]

        if self.auto.load:
            self.load(
                series_id=self.series_id,
                start_date=self.start_date,
                end_date=self.end_date,
                cycle=self.cycle,
            )

    def get(self, series_id, start_date=None, end_date=None, cycle="", **kwargs):
        codes = [None, None, None, None, None]
        if "/" in series_id:
            series_id = series_id.split("/")
            codes[: len(series_id)] = series_id
        else:
            codes[0] = series_id
        data, _ = self.get_ecos(
            service="StatisticSearch",
            stat_code=codes[0],
            cycle=cycle,
            start_date=start_date,
            end_date=end_date,
            item_code1=codes[1],
            item_code2=codes[2],
            item_code3=codes[3],
            item_code4=codes[4],
            **kwargs,
        )
        return data

    def get_ecos(
        self,
        service,
        stat_code,
        cycle="",
        start_date="",
        end_date="",
        start_num=None,
        end_num=None,
        item_code1=None,
        item_code2=None,
        item_code3=None,
        item_code4=None,
        lang=None,
        format=None,
        **kwargs,
    ):
        import requests

        lang = lang or self._api.lang
        format = format or self._api.format
        start_num = start_num or self._api.start_num
        end_num = end_num or self._api.end_num

        url = "{}/{}/{}/{}/{}/{}/{}/{}/{}/{}/{}".format(
            self._api.base_url,
            service,
            self._ecos_api_key,
            format,
            lang,
            start_num,
            end_num,
            stat_code,
            cycle,
            start_date,
            end_date,
        )
        if item_code1 is not None:
            item_code2 = item_code2 or "?"
            item_code3 = item_code3 or "?"
            item_code4 = item_code4 or "?"

            url += "/{}/{}/{}/{}".format(item_code1, item_code2, item_code3, item_code4)

        response = requests.get(url).content.decode("utf-8")
        response = json.loads(response)
        if service in response:
            data = pd.DataFrame(response[service]["row"])
            num_rows = response[service]["list_total_count"]

            return data, num_rows
        else:
            log.info(f"No data found for {service}")
            if self.verbose:
                print(response)
            return None, None

    def load(
        self,
        series_id,
        start_date=None,
        end_date=None,
        cycle=None,
        filename=None,
        expressions=None,
        reset_index=False,
    ):
        if isinstance(series_id, str):
            series_id = [series_id]
        elif not isinstance(series_id, list):
            series_id = []

        if filename is None:
            file_name = "_".join(series_id).replace("/", "_")
            filename = f"{file_name}.parquet"

        if start_date is None:
            start_date = self.start_date
        if end_date is None:
            end_date = self.end_date
        if cycle is None:
            cycle = self.cycle

        if isinstance(self.eval_columns, dict):
            self.eval_columns["expressions"] = expressions

        if self.verbose:
            print(f"Loading {series_id} from {start_date} to {end_date}")

        filepath = os.path.join(self.output_dir, filename)
        if not os.path.exists(filepath) or self.force.download:
            self._data = self._load_series(
                series_id, start_date, end_date, cycle, reset_index
            )
            eKonf.save_data(self.data, filepath, verbose=self.verbose)
        else:
            log.info(f"{filepath} already exists.")
            self._data = eKonf.load_data(filepath, verbose=self.verbose)
        return self.data.copy()

    def _load_series(
        self,
        series_ids=None,
        start_date=None,
        end_date=None,
        cycle=None,
        reset_index=False,
        **kwargs,
    ):

        _dfs = []
        for series_id in series_ids:
            df = self.get(
                series_id=series_id,
                start_date=start_date,
                end_date=end_date,
                cycle=cycle,
                **kwargs,
            )
            df = eKonf.pipe(df, self.eval_columns)
            if len(series_ids) > 1:
                df["series_id"] = series_id
            if self.verbose:
                print(df.head())
            _dfs.append(df)

        df = pd.concat(_dfs)
        if isinstance(reset_index, str) and reset_index in df.columns:
            df = df.set_index(reset_index)
        elif reset_index:
            df = df.reset_index()
        return df
