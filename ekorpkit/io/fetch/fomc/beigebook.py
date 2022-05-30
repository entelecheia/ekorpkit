import sys
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# Import parent class
from .base import FOMC


class BeigeBook(FOMC):
    """
    A convenient class for extracting statement from the FOMC website
    """

    def __init__(self, content_type, **args):
        super().__init__(content_type, **args)
        self.beige_book_url = (
            "https://www.federalreserve.gov/monetarypolicy/beige-book-default.htm"
        )
        # Please refer to https://www.federalreserve.gov/monetarypolicy/beige-book-archive.htm for the start year

    def _get_links(self, from_year):
        """
        Override private function that sets all the links for the contents to download on FOMC website
        from from_year (=min(2015, from_year)) to the current most recent year
        """
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        r = requests.get(self.beige_book_url)
        soup = BeautifulSoup(r.text, "html.parser")

        # Getting links from current page. Meetin scripts are not available.
        if self.verbose:
            print("Getting links for Beigebooks...")
        contents = soup.find_all(
            "a", href=re.compile("^/monetarypolicy/beigebook\d{6}.htm")
        )
        dates = soup.find_all(
            "a",
            href=re.compile("/monetarypolicy/files/BeigeBook_\d{8}.pdf"),
        )
        dates = [content.attrs["href"] for content in dates]
        self.links = [content.attrs["href"] for content in contents]
        self.speakers = [
            self._speaker_from_date(self._date_from_link(x)) for x in dates
        ]
        self.titles = ["FOMC Beige Book"] * len(self.links)
        self.dates = [
            datetime.strptime(self._date_from_link(x), "%Y-%m-%d") for x in dates
        ]
        # TODO: _date_from_link deos not work for beigebooks, find date from content instead
        # Correct some date in the link does not match with the meeting date
        for i, m_date in enumerate(self.dates):
            if m_date == datetime(2019, 10, 11):
                self.dates[i] = datetime(2019, 10, 4)

        if self.verbose:
            print(
                "{} links and {} dates found in the current page.".format(
                    len(self.links), len(self.dates)
                )
            )
        if from_year <= 1995:
            print("Archive only from 1996, so setting from_year as 1996...")
            from_year = 1996

        to_year = int(datetime.today().strftime("%Y"))
        if from_year <= to_year:
            print("Getting links from archive pages...")
            for year in range(from_year, to_year + 1):
                yearly_contents = []
                beige_book_annual_url = (
                    self.base_url + "/monetarypolicy/beigebook" + str(year) + ".htm"
                )
                r_year = requests.get(beige_book_annual_url)
                soup_yearly = BeautifulSoup(r_year.text, "html.parser")
                yearly_contents = soup_yearly.find_all("a", text="HTML")
                yearly_dates = soup.find_all("a", text="PDF")
                yearly_dates = [content.attrs["href"] for content in yearly_dates]
                for yearly_content, yearly_date in zip(yearly_contents, yearly_dates):
                    self.links.append(yearly_content.attrs["href"])
                    self.speakers.append(
                        self._speaker_from_date(self._date_from_link(yearly_date))
                    )
                    self.titles.append("FOMC Beige Book")
                    self.dates.append(
                        datetime.strptime(self._date_from_link(yearly_date), "%Y-%m-%d")
                    )
                    # Correct some date in the link does not match with the meeting date
                    if self.dates[-1] == datetime(2007, 6, 18):
                        self.dates[-1] = datetime(2007, 6, 28)
                    elif self.dates[-1] == datetime(2007, 8, 17):
                        self.dates[-1] = datetime(2007, 8, 16)
                    elif self.dates[-1] == datetime(2008, 1, 22):
                        self.dates[-1] = datetime(2008, 1, 21)
                    elif self.dates[-1] == datetime(2008, 3, 11):
                        self.dates[-1] = datetime(2008, 3, 10)
                    elif self.dates[-1] == datetime(2008, 10, 8):
                        self.dates[-1] = datetime(2008, 10, 7)
                    if self.verbose:
                        print(f'{self.dates[-1]}: {self.links[-1]}')

                if self.verbose:
                    print(
                        "YEAR: {} - {} links found.".format(year, len(yearly_contents))
                    )
        print("There are total ", len(self.links), " links for ", self.content_type)

    def _add_article(self, link, index=None):
        """
        Override a private function that adds a related article for 1 link into the instance variable
        The index is the index in the article to add to.
        Due to concurrent processing, we need to make sure the articles are stored in the right order
        """
        if self.verbose:
            sys.stdout.write(".")
            sys.stdout.flush()

        link = self.base_url + link.replace(self.base_url, "")
        print(link)
        res = requests.get(link)
        html = res.text
        article = BeautifulSoup(html, "html.parser")
        paragraphs = article.findAll("p")
        para_text = []
        for paragraph in paragraphs:
            text = re.sub(r"\s+", " ", paragraph.get_text().strip())
            if len(text) > 0 and text not in self.skip_text_list:
                para_text.append(text)
        self.articles[index] = self.segment_separator.join(para_text)
