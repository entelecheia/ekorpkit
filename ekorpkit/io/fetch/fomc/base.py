import logging
import re
import threading
import os
import numpy as np
import pandas as pd
import codecs
import requests
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
from datetime import datetime
from abc import abstractmethod
from ekorpkit import eKonf


log = logging.getLogger(__name__)


def download_data(fomc, from_year, verbose=False):
    if not os.path.exists(fomc.output_filepath) or fomc.force_download:
        log.info(f"Downloading {fomc.content_type}")
        df = fomc.get_contents(from_year)
        log.info("Shape of the downloaded data: ", df.shape)
        if verbose:
            print("The first 5 rows of the data: \n", df.head())
            print("The last 5 rows of the data: \n", df.tail())
        fomc.save()
    else:
        log.info(f"{fomc.content_type} already exists")


def build_fomc(**args):

    args = eKonf.to_config(args)
    from_year = args.from_year
    verbose = args.verbose
    if (from_year < 1980) or (from_year > 2020):
        log.warning("Please specify the second argument between 1980 and 2020")
        return

    for content in args.contents:
        fomc = eKonf.instantiate(content, **args.fomc)
        download_data(fomc, from_year, verbose)


class FOMC:
    """
    A base class for extracting documents from the FOMC website
    """

    def __init__(self, content_type, **args):
        args = eKonf.to_dict(args)
        self.verbose = args["verbose"]
        log.info(f"Initializing {content_type}")

        # Set arguments to internal variables
        self.content_type = content_type
        self.num_workers = args["num_workers"]
        self.output_dir = args["output_dir"]
        self.output_raw_dir = os.path.join(self.output_dir, "raw")
        os.makedirs(self.output_raw_dir, exist_ok=True)
        self.output_file = content_type + ".parquet"
        self.output_filepath = os.path.join(self.output_dir, self.output_file)
        self.calendar_filepath = os.path.join(self.output_dir, "fomc_calendar.parquet")

        self.segment_separator = codecs.decode(
            args["segment_separator"], "unicode_escape"
        )
        self.force_download = args["force_download"]

        # Initialization
        self.df = None
        self.links = None
        self.dates = None
        self.articles = None
        self.speakers = None
        self.titles = None

        # missing articles
        self._articles = args.get("articles")

        # FOMC website URLs
        self.base_url = args["base_url"]
        self.calendar_url = args["calendar_url"]

        _meta = args["meta"]
        # FOMC Chairperson's list
        self.chairpersons = _meta.get("chairpersons")
        if isinstance(self.chairpersons, dict):
            self.chairpersons = pd.DataFrame(self.chairpersons.values())
        self.canlendar = None

        self.econ_series = _meta.get("econ_series")
        self.recessions = _meta.get("recessions")
        if isinstance(self.recessions, dict):
            self.recessions = pd.DataFrame(self.recessions.values())

        self.unconventionals = _meta.get("unconventionals")
        if isinstance(self.unconventionals, dict):
            self.unconventionals = pd.DataFrame(self.unconventionals.values())
            self.unconventionals["speaker"] = self.unconventionals.date.map(
                self._speaker_from_date
            )

        # skip list
        self.skip_text_list = [
            "Return to top",
        ]

    def _date_from_link(self, link):
        # TODO: fix this
        date = re.findall("[0-9]{8}", link)[0]
        if date[4] == "0":
            date = "{}-{}-{}".format(date[:4], date[5:6], date[6:])
        else:
            date = "{}-{}-{}".format(date[:4], date[4:6], date[6:])
        return date

    def _speaker_from_date(self, article_date):
        """
        Returns the speaker of the article based on the date of the article
        """
        if isinstance(article_date, str):
            article_date = datetime.strptime(article_date, "%Y-%m-%d")

        speaker = self.chairpersons.query(
            "from_date < @article_date & to_date > @article_date"
        )
        if speaker.empty:
            return "other"
        else:
            speaker = speaker.iloc[0]
            return speaker.first_name + " " + speaker.last_name

    @abstractmethod
    def _get_links(self, from_year):
        """
        private function that sets all the links for the FOMC meetings
        from the giving from_year to the current most recent year
        from_year is min(2015, from_year)
        """
        # Implement in sub classes
        # TODO - replace links list with dictionary with key as date
        pass

    @abstractmethod
    def _add_article(self, link, index=None):
        """
        adds the related article for 1 link into the instance variable
        index is the index in the article to add to. Due to concurrent
        prcessing, we need to make sure the articles are stored in the
        right order
        """
        # Implement in sub classes
        pass

    def _get_articles_multi(self):
        """
        gets all articles using multi-threading
        """
        if self.verbose:
            print("Getting articles - Multi-threaded...")

        self.articles = [""] * len(self.links)
        jobs = []
        # initiate and start threads:
        index = 0
        while index < len(self.links):
            if len(jobs) < self.num_workers:
                t = threading.Thread(
                    target=self._add_article,
                    args=(
                        self.links[index],
                        index,
                    ),
                )
                jobs.append(t)
                t.start()
                index += 1
            else:  # wait for threads to complete and join them back into the main thread
                t = jobs.pop(0)
                t.join()
        for t in jobs:
            t.join()

        # for row in range(len(self.articles)):
        #    self.articles[row] = self.articles[row].strip()

    def get_contents(self, from_year=1990):
        """
        Returns a Pandas DataFrame with the date as the index for a date range of from_year to the most current.
        Save the same to internal df as well.
        """
        self._get_links(from_year)
        self._get_articles_multi()
        dict = {
            "date": self.dates,
            "speaker": self.speakers,
            "title": self.titles,
            "text": self.articles,
        }
        self.df = pd.DataFrame(dict)
        self.df["content_type"] = self.content_type
        _df = self.get_missing_contents(self.content_type)
        if _df is not None:
            self.df = pd.concat([self.df, _df])
        self.df.sort_values(by="date", inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def get_missing_contents(self, content_type):
        if isinstance(self._articles, dict) and content_type in self._articles:
            articles = self._articles[content_type]
            df = pd.DataFrame.from_dict(articles, orient="records")
            df["content_type"] = content_type
            return df
        else:
            return None

    def save(self):
        """
        save the dataframe to a parquet file
        """
        log.info("Writing to ", self.output_filepath)
        eKonf.save_data(self.df, self.output_filepath)

    def load_calendar(self, from_year=None, force_download=False):
        """
        get the calendar from the FOMC website
        """
        if os.path.exists(self.calendar_filepath) and not force_download:
            log.info("Loading calendar from cache...")
            self.calendar = eKonf.load_data(self.calendar_filepath)
            return self.calendar

        if self.verbose:
            print("Getting calendar...")

        if from_year:
            from_year = int(from_year)

            if (from_year < 1936) or (from_year > 2017):
                print("Please specify the first argument between 1936 and 2015")
                return
        else:
            from_year = 1936
            print(
                "From year is set as 1936. Please specify the year as the first argument if required."
            )

        fomc_dates = []
        # Retrieve FOMC Meeting date from current page - recent five years
        r = requests.get(self.calendar_url)
        soup = BeautifulSoup(r.text, "html.parser")
        panel_divs = soup.find_all("div", {"class": "panel panel-default"})

        for panel_div in panel_divs:
            m_year = panel_div.find("h4").get_text()[:4]
            m_months = panel_div.find_all("div", {"class": "fomc-meeting__month"})
            m_dates = panel_div.find_all("div", {"class": "fomc-meeting__date"})
            if self.verbose:
                print("YEAR: {} - {} meetings found.".format(m_year, len(m_dates)))

            for (m_month, m_date) in zip(m_months, m_dates):
                month_name = m_month.get_text().strip()
                date_text = m_date.get_text().strip()
                is_forecast = False
                is_unscheduled = False
                is_month_short = False

                if "cancelled" in date_text:
                    continue
                elif "notation vote" in date_text:
                    date_text = date_text.replace("(notation vote)", "").strip()
                elif "unscheduled" in date_text:
                    date_text = date_text.replace("(unscheduled)", "").strip()
                    is_unscheduled = True

                if "*" in date_text:
                    date_text = date_text.replace("*", "").strip()
                    is_forecast = True

                if "/" in month_name:
                    month_name = re.findall(r".+/(.+)$", month_name)[0]
                    is_month_short = True

                if "-" in date_text:
                    date_text = re.findall(r".+-(.+)$", date_text)[0]

                meeting_date_str = m_year + "-" + month_name + "-" + date_text
                if is_month_short:
                    meeting_date = datetime.strptime(meeting_date_str, "%Y-%b-%d")
                else:
                    meeting_date = datetime.strptime(meeting_date_str, "%Y-%B-%d")

                fomc_dates.append(
                    {
                        "date": meeting_date,
                        "unscheduled": is_unscheduled,
                        "forecast": is_forecast,
                        "confcall": False,
                    }
                )
        min_recent_year = min(fomc_dates, key=lambda x: x["date"]).get("date").year
        # Retrieve FOMC Meeting date older than 2015
        for year in range(from_year, min_recent_year):
            hist_url = self.base_url + f"/monetarypolicy/fomchistorical{year}.htm"
            r = requests.get(hist_url)
            soup = BeautifulSoup(r.text, "html.parser")
            if year >= 2011:
                panel_headings = soup.find_all("h5", {"class": "panel-heading"})
            else:
                panel_headings = soup.find_all("div", {"class": "panel-heading"})
            log.info("YEAR: {} - {} meetings found.".format(year, len(panel_headings)))
            for panel_heading in panel_headings:
                date_text = panel_heading.get_text().strip()
                # print("Date: ", date_text)
                regex = r"(January|February|March|April|May|June|July|August|September|October|November|December).*\s(\d*-)*(\d+)\s+(Meeting|Conference Calls?|\(unscheduled\))\s-\s(\d+)"
                date_text_ext = re.findall(regex, date_text)[0]
                meeting_date_str = (
                    date_text_ext[4] + "-" + date_text_ext[0] + "-" + date_text_ext[2]
                )
                # print("   Extracted:", meeting_date_str)
                if meeting_date_str == "1992-June-1":
                    meeting_date_str = "1992-July-1"
                elif meeting_date_str == "1995-January-1":
                    meeting_date_str = "1995-February-1"
                elif meeting_date_str == "1998-June-1":
                    meeting_date_str = "1998-July-1"
                elif meeting_date_str == "2012-July-1":
                    meeting_date_str = "2012-August-1"
                elif meeting_date_str == "2013-April-1":
                    meeting_date_str = "2013-May-1"

                meeting_date = datetime.strptime(meeting_date_str, "%Y-%B-%d")
                is_confcall = "Conference Call" in date_text_ext[3]
                is_unscheduled = "unscheduled" in date_text_ext[3]
                fomc_dates.append(
                    {
                        "date": meeting_date,
                        "unscheduled": is_unscheduled,
                        "forecast": False,
                        "confcall": is_confcall,
                    }
                )

        df = pd.DataFrame(fomc_dates).sort_values(by=["date"])

        df["speaker"] = df.date.map(self._speaker_from_date)

        # # Use date as index
        df.set_index("date", inplace=True)

        eKonf.save_data(df, self.calendar_filepath)
        self.calendar = df
        return df.copy()

    def add_decisions_to_calendar(self, fedrates, series_id="DFEDTAR"):
        """
        The target range was changed a couple of days after the announcement in the past,
         while it is immediately put in effect on the day recently.
        Use the target rate three days after the meeting as target announced,
         compare it with previous day's rate to check if rate has been changed.
          -1: Rate lower
           0: No change
          +1: Rate hike
        """
        import numpy as np
        from tqdm.auto import tqdm

        # fedrates = fedrates.copy()[fedrates.index >= self.calendar.index.min()]
        fedrates["rate"] = fedrates[series_id].shift(-3)
        fedrates["prv_rate"] = fedrates[series_id].shift(1)
        fedrates["rate_change"] = fedrates["rate"] - fedrates["prv_rate"]
        fedrates["rate_decision"] = np.where(
            fedrates["rate_change"] > 0, 1, np.where(fedrates["rate_change"] < 0, -1, 0)
        )
        index_name = "index"
        columns = ["rate", "rate_change", "rate_decision"]
        fomc_calendar = self.calendar.copy()
        results = []
        for i, row in tqdm(fomc_calendar.iterrows(), total=fomc_calendar.shape[0]):
            src_ix = fedrates[fedrates.index == row.name].index.min()
            if not pd.isnull(src_ix):
                data = fedrates.loc[src_ix, columns].to_dict()
                data[index_name] = row.name
                results.append(data)
            else:
                data = {col: None for col in columns}
                data[index_name] = row.name
                results.append(data)

        results = pd.DataFrame(results)
        results.set_index(index_name, inplace=True)
        fomc_calendar[columns] = results[columns]
        fomc_calendar = fomc_calendar[fomc_calendar["rate"].notnull()]

        eKonf.save_data(fomc_calendar, self.calendar_filepath)
        self.calendar = fomc_calendar
        return fomc_calendar.copy()

    def add_unconventionals_to_calendar(self):
        """
        Add the unconventionals to the calendar.
          -2: Unconventional easing
          -1: Rate lower
           0: No change
          +1: Rate hike
          +2: Unconventional tightening
        """

        fomc_calendar = self.calendar.copy()
        unconventionals = self.unconventionals.copy().set_index("date")
        columns = ["unscheduled", "speaker", "rate", "rate_decision", "rate_change"]

        for ix in unconventionals.index:
            fomc_calendar.loc[ix, columns] = unconventionals.loc[ix, columns]

        columns = ["forecast", "confcall"]
        fomc_calendar[columns] = fomc_calendar[columns].fillna(False)
        fomc_calendar.sort_index(ascending=True, inplace=True)
        fomc_calendar["rate_changed"] = fomc_calendar["rate_decision"].apply(
            lambda x: 0 if x == 0 else 1
        )

        eKonf.save_data(fomc_calendar, self.calendar_filepath)
        self.calendar = fomc_calendar
        return fomc_calendar.copy()

    @staticmethod
    def add_available_latest(target, source, name, columns, date_offset, window=1):
        from dateutil.relativedelta import relativedelta

        date_offset = relativedelta(**date_offset)
        index_name = target.index.name
        source_ma = source.rolling(window).mean()

        results = []
        for i, row_tgt in tqdm(target.iterrows(), total=target.shape[0]):
            src_ix = source_ma[source_ma.index < row_tgt.name - date_offset].index.max()
            if not pd.isnull(src_ix):
                data = source_ma.loc[src_ix, columns].to_dict()
                data[name + "_date"] = source_ma.loc[src_ix].name
                data[index_name] = row_tgt.name
                results.append(data)
            else:
                data = {col: None for col in columns}
                data[name + "_date"] = None
                data[index_name] = row_tgt.name
                results.append(data)
        if target.shape[0] != len(results):
            print(
                "target has {} rows but returned {} rows from source!".format(
                    target.shape[0], len(results)
                )
            )
        results = pd.DataFrame(results)
        results.set_index(index_name, inplace=True)
        return target.merge(results, how="left", left_index=True, right_index=True)

    def find_speaker_of_sections(self, data):
        """
        Find the speaker of each section.
        """

        def find_speaker(row):
            content_type = row.content_type
            speaker = row.speaker
            text = row.text
            if content_type in ["fomc_press_conf", "fomc_meeting_script"]:
                match = re.findall(r"(^[A-Za-zŞ. ]*[A-Z]{3}).\d? (.*)", text)
                if len(match) == 0:
                    match = re.findall(r"(^[A-Za-zŞ. ]*[A-Z]{3}).\d(.*)", text)
                if len(match) == 1:
                    speaker, text = match[0]
                    return speaker
                return None
            else:
                return speaker

        df = data.copy()
        df["speaker"] = df.apply(find_speaker, axis=1)
        return df

    def postprocess_metadata(self, data):
        """
        - Add type
        - Add rate, decision (for meeting documents, None for the others)
        - Add next meeting date, rate and decision
        """

        def is_meeting_doc(content_type):
            if content_type in (
                "fomc_statement",
                "fomc_minutes",
                "fomc_press_conf",
                "fomc_meeting_script",
            ):
                return True
            elif content_type in ("fomc_speech", "fomc_testimony", "fomc_beigebook"):
                return False
            else:
                log.warning(f"Invalid doc_type [{content_type}] is given!")
                return None

        df = data.copy()

        df["decision"] = df.apply(
            lambda x: self._get_rate_change(x["date"])
            if is_meeting_doc(x["content_type"])
            else None,
            axis=1,
        )
        df["rate"] = df.apply(
            lambda x: self._get_rate(x["date"])
            if is_meeting_doc(x["content_type"])
            else None,
            axis=1,
        )
        df["recent_meeting"] = df["date"].map(
            lambda x: self._get_recent_meeting_date(x)
        )
        df["recent_decision"] = df["recent_meeting"].map(
            lambda x: self._get_rate_change(x)
        )
        df["recent_rate"] = df["recent_meeting"].map(lambda x: self._get_rate(x))
        df["next_meeting"] = df["date"].map(lambda x: self._get_next_meeting_date(x))
        df["next_decision"] = df["next_meeting"].map(lambda x: self._get_rate_change(x))
        df["next_rate"] = df["next_meeting"].map(lambda x: self._get_rate(x))

        return df

    def _get_rate_change(self, article_date):
        """
        Returns rate change decision of the FOMC Decision for the given date x.
        x should be of datetime type or yyyy-mm-dd format string.
        """
        if isinstance(article_date, str):
            article_date = datetime.strptime(article_date, "%Y-%m-%d")

        if article_date in self.calendar.index:
            return self.calendar.loc[article_date]["rate_decision"]
        else:
            return None

    def _get_rate(self, article_date):
        """
        Returns rate of the FOMC Decision for the given date x.
        x should be of datetime type or yyyy-mm-dd format string.
        """
        if isinstance(article_date, str):
            article_date = datetime.strptime(article_date, "%Y-%m-%d")

        if article_date in self.calendar.index:
            return self.calendar.loc[article_date]["rate"]
        else:
            return None

    def _get_next_meeting_date(self, article_date):
        """
        Returns the next fomc meeting date for the given date x, referring to fomc_calendar DataFrame.
        Usually FOMC Meetings takes two days, so it starts searching from x+2.
        x should be of datetime type or yyyy-mm-dd format string.
        """
        import datetime as dt

        if isinstance(article_date, str):
            article_date = datetime.strptime(article_date, "%Y-%m-%d")

        # Add two days to get the day after next
        article_date += dt.timedelta(days=2)

        cal = self.calendar.copy()
        if cal.index.min() > article_date:
            # If the date is older than the first FOMC Meeting, do not return any date.
            return None
        else:
            dt_ix = cal[cal.index > article_date].index.min()
            if not pd.isnull(dt_ix):
                return dt_ix
            else:
                return None

    def _get_recent_meeting_date(self, article_date):
        """
        Returns the most recent fomc meeting date for the given date x, referring to fomc_calendar DataFrame.
        Usually FOMC Meetings takes two days, so it starts searching from x+2.
        x should be of datetime type or yyyy-mm-dd format string.
        """
        import datetime as dt

        if isinstance(article_date, str):
            article_date = datetime.strptime(article_date, "%Y-%m-%d")

        # Add two days to get the day after next
        article_date += dt.timedelta(days=2)

        cal = self.calendar.copy()
        if cal.index.min() > article_date:
            # If the date is older than the first FOMC Meeting, do not return any date.
            return None
        else:
            dt_ix = cal[cal.index < article_date].index.max()
            if not pd.isnull(dt_ix):
                return dt_ix
            else:
                return None

    @staticmethod
    def get_irf(data, nlags=4, nirf=20):
        import statsmodels.api as sm

        # Dimensions
        nobs = data.shape[0] - nlags
        neqs = data.shape[1]

        # Estimate via statsmodels VAR
        mod = sm.tsa.VAR(data)
        res = mod.fit(maxlags=nlags, ic=None, trend="n")

        # Variance / Covariance matrix
        omega = res.sigma_u

        # Take LDL decomposition of the variance / covariance matrix
        L = np.linalg.cholesky(omega)
        D = np.diag(L.diagonal() ** 2)
        L[np.diag_indices(L.shape[0])] = 1

        # Storage
        irf_tilde = np.zeros((nirf + nlags + 1, neqs, neqs))
        irf = np.zeros((nirf, neqs, neqs))

        # Initial values
        irf_tilde[nlags] = np.eye(neqs)
        irf[0] = L

        # Iteratively create IRFs
        for i in range(nirf):
            for j in range(nlags):
                irf_tilde[i + nlags + 1] += np.dot(
                    res.params.iloc[j * neqs : (j + 1) * neqs, :].T,
                    irf_tilde[i + nlags - j],
                )
            irf[i] = np.dot(irf_tilde[i + nlags], L)

        irfs = {}
        for col in data.columns:
            ix = data.columns.get_loc(col)
            irf_df = pd.DataFrame(-irf[:, :, ix])
            irf_df.columns = data.columns
            irfs[col] = irf_df

        return irfs

    @staticmethod
    def plot_irf(irfs, impulse_name, ncols=2, figsize=(16, 12), title=None):
        irf = irfs[impulse_name]

        title = title or f"Impulse Response of {impulse_name} shock"
        names = irf.columns.tolist()
        neqs = len(names)

        cfg = eKonf.compose("visualize/plot=lineplot")

        cfg.figure.figsize = figsize
        cfg.subplots.ncols = ncols
        cfg.subplots.nrows = int(np.ceil(neqs / ncols))
        lineplot = cfg.lineplot.copy()
        ax = cfg.ax.copy()
        cfg.plots = []
        cfg.axes = []

        for i, name in enumerate(names):
            plot = lineplot.copy()
            plot.y = name
            plot.axno = i
            cfg.plots.append(plot)
            _ax = ax.copy()
            _ax.title = name
            _ax.axno = i
            cfg.axes.append(_ax)
        cfg.figure.super.title = title

        eKonf.instantiate(cfg, data=irf)
