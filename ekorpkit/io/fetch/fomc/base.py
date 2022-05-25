import re
import threading
import os
import pandas as pd
import codecs
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from abc import abstractmethod
from ekorpkit import eKonf
from ekorpkit.io.file import save_dataframe, load_dataframe


def download_data(fomc, from_year):
    if not os.path.exists(fomc.output_filepath) or fomc.force_download:
        print(f"Downloading {fomc.content_type}")
        df = fomc.get_contents(from_year)
        print("Shape of the downloaded data: ", df.shape)
        print("The first 5 rows of the data: \n", df.head())
        print("The last 5 rows of the data: \n", df.tail())
        fomc.save()
    else:
        print(f"{fomc.content_type} already exists")


def build_fomc(**args):

    args = eKonf.to_config(args)
    from_year = args.from_year
    if (from_year < 1980) or (from_year > 2020):
        print("Please specify the second argument between 1980 and 2020")
        return

    for content in args.contents:
        fomc = eKonf.instantiate(content, **args.fomc)
        download_data(fomc, from_year)


class FOMC:
    """
    A base class for extracting documents from the FOMC website
    """

    def __init__(self, content_type, **args):
        args = eKonf.to_dict(args)
        self.verbose = args["verbose"]
        if self.verbose:
            print(content_type)
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

        # FOMC website URLs
        self.base_url = args["base_url"]
        self.calendar_url = args["calendar_url"]

        # FOMC Chairperson's list
        self.chair = pd.DataFrame(args["chair"])
        self.chair["from_date"] = pd.to_datetime(self.chair.from_date)
        self.chair["to_date"] = pd.to_datetime(self.chair.to_date)
        self.canlendar = None

        self.econ_series = args["econ_series"]
        self.recessions = pd.DataFrame(args["recessions"])
        _format = self.recessions.format[0]
        self.recessions.drop(columns="format", inplace=True)
        self.recessions["from_date"] = pd.to_datetime(
            self.recessions.from_date, format=_format
        )
        self.recessions["to_date"] = pd.to_datetime(
            self.recessions.to_date, format=_format
        )
        self.unconventionals = pd.DataFrame(args["unconventionals"])
        _format = self.unconventionals.format[0]
        self.unconventionals.drop(columns="format", inplace=True)
        self.unconventionals["date"] = pd.to_datetime(
            self.unconventionals.date, format=_format
        )
        self.unconventionals["chair"] = self.unconventionals.date.map(
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

        speaker = self.chair.query(
            "from_date < @article_date & to_date > @article_date"
        )
        if speaker.empty:
            return "other"
        else:
            speaker = speaker.iloc[0]
            return speaker.first_name + " " + speaker.surname

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
        self.df = pd.DataFrame(dict).sort_values(by=["date"])
        self.df["content_type"] = self.content_type
        self.df.reset_index(drop=True, inplace=True)
        return self.df

    def save(self):
        """
        save the dataframe to a csv file
        """
        if self.verbose:
            print("Writing to ", self.output_filepath)
        os.makedirs(self.output_dir, exist_ok=True)
        save_dataframe(self.df, self.output_filepath)

    def load_calendar(self, from_year=None, force_download=False):
        """
        get the calendar from the FOMC website
        """
        if os.path.exists(self.calendar_filepath) and not force_download:
            if self.verbose:
                print("Loading calendar from cache...")
            self.calendar = load_dataframe(self.calendar_filepath)
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
            if self.verbose:
                print("YEAR: {} - {} meetings found.".format(year, len(panel_headings)))
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

        df["chair"] = df.date.map(self._speaker_from_date)

        # # Use date as index
        df.set_index("date", inplace=True)

        save_dataframe(df, self.calendar_filepath)
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
        from tqdm import tqdm

        rate_list = []
        decision_list = []
        rate_change_list = []
        fedrates = fedrates.copy()[fedrates.index >= self.calendar.index.min()]
        fomc_calendar = self.calendar.copy()
        for i in tqdm(range(len(fomc_calendar))):
            not_found = True
            for j in range(len(fedrates)):
                if fomc_calendar.index[i] == fedrates.index[j]:
                    not_found = False
                    rate_list.append(float(fedrates[series_id].iloc[j + 3]))
                    rate_change_list.append(
                        float(fedrates[series_id].iloc[j + 3])
                        - float(fedrates[series_id].iloc[j - 1])
                    )
                    if (
                        fedrates[series_id].iloc[j - 1]
                        == fedrates[series_id].iloc[j + 3]
                    ):
                        decision_list.append(0)
                    elif (
                        fedrates[series_id].iloc[j - 1]
                        < fedrates[series_id].iloc[j + 3]
                    ):
                        decision_list.append(1)
                    elif (
                        fedrates[series_id].iloc[j - 1]
                        > fedrates[series_id].iloc[j + 3]
                    ):
                        decision_list.append(-1)
                    break
            if not_found:
                rate_list.append(np.nan)
                decision_list.append(np.nan)
                rate_change_list.append(np.nan)

        fomc_calendar["rate"] = rate_list
        fomc_calendar["rate_change"] = rate_change_list
        fomc_calendar["rate_decision"] = decision_list
        fomc_calendar["rate_decision"] = fomc_calendar["rate_decision"].astype("Int8")
        fomc_calendar = fomc_calendar[fomc_calendar["rate"].notnull()]

        save_dataframe(fomc_calendar, self.calendar_filepath)
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
        columns = ["unscheduled", "chair", "rate", "rate_decision", "rate_change"]

        for ix in unconventionals.index:
            fomc_calendar.loc[ix, columns] = unconventionals.loc[ix, columns]

        columns = ["forecast", "confcall"]
        fomc_calendar[columns] = fomc_calendar[columns].fillna(False)
        fomc_calendar.sort_index(ascending=True, inplace=True)
        fomc_calendar["rate_changed"] = fomc_calendar["rate_decision"].apply(
            lambda x: 0 if x == 0 else 1
        )

        save_dataframe(fomc_calendar, self.calendar_filepath)
        self.calendar = fomc_calendar
        return fomc_calendar.copy()
