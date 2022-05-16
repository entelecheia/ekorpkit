import re
import threading
import os
import pandas as pd
import codecs
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf


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
    from hydra.utils import instantiate

    args = eKonf.to_config(args)
    from_year = args.from_year
    if (from_year < 1980) or (from_year > 2020):
        print("Please specify the second argument between 1980 and 2020")
        return

    for content in args.contents:
        fomc = instantiate(content, **args.fomc, _recursive_=False)
        download_data(fomc, from_year)


class FomcBase(metaclass=ABCMeta):
    """
    A base class for extracting documents from the FOMC website
    """

    def __init__(self, content_type, **args):
        args = eKonf.to_config(args)
        print(content_type)
        # Set arguments to internal variables
        self.content_type = content_type
        self.verbose = args.verbose
        self.num_workers = args.num_workers
        self.output_dir = args.output_dir
        self.output_raw_dir = args.output_dir + "/raw"
        os.makedirs(self.output_raw_dir, exist_ok=True)
        self.output_file = content_type + (".csv.bz2" if args.compress else ".csv")
        self.output_filepath = self.output_dir + "/" + self.output_file

        self.segment_separator = codecs.decode(args.segment_separator, "unicode_escape")
        self.force_download = args.force_download

        # Initialization
        self.df = None
        self.links = None
        self.dates = None
        self.articles = None
        self.speakers = None
        self.titles = None

        # FOMC website URLs
        self.base_url = args.base_url
        self.calendar_url = args.calendar_url

        # FOMC Chairperson's list
        self.chair = pd.DataFrame(data=args.chair.data, columns=args.chair.columns)

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
        if (
            self.chair.from_date[0] < article_date
            and article_date < self.chair.to_date[0]
        ):
            speaker = self.chair.first_name[0] + " " + self.chair.surname[0]
        elif (
            self.chair.from_date[1] < article_date
            and article_date < self.chair.to_date[1]
        ):
            speaker = self.chair.first_name[1] + " " + self.chair.surname[1]
        elif (
            self.chair.from_date[2] < article_date
            and article_date < self.chair.to_date[2]
        ):
            speaker = self.chair.first_name[2] + " " + self.chair.surname[2]
        elif (
            self.chair.from_date[3] < article_date
            and article_date < self.chair.to_date[3]
        ):
            speaker = self.chair.first_name[3] + " " + self.chair.surname[3]
        else:
            speaker = "other"
        return speaker

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
        self.df.to_csv(self.output_filepath, index=False)
