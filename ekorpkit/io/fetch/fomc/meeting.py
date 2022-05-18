import os
import sys
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pdfplumber

# Import parent class
from .base import FOMC


class MeetingScript(FOMC):
    """
    A convenient class for extracting meeting scripts from the FOMC website.
    FOMC publishes the meeting scripts after 5 years, so this cannot be used for the prediction of the monetary policy in real-time.
    """

    def __init__(self, content_type, **args):
        super().__init__(content_type, **args)
        left_pct = 0.05  # % Distance of left side of character from left side of page.
        top_pct = 0.10  # % Distance of top of character from top of page.
        right_pct = (
            0.95  # % Distance of right side of character from left side of page.
        )
        bottom_pct = 0.88  #% Distance of bottom of the character from top of page.
        self.crop_coords = [left_pct, top_pct, right_pct, bottom_pct]

    def _get_links(self, from_year):
        """
        Override private function that sets all the links for the contents to download on FOMC website
         from from_year (=min(2015, from_year)) to the current most recent year
        """
        self.links = []
        self.titles = []
        self.speakers = []
        self.dates = []

        r = requests.get(self.calendar_url)
        soup = BeautifulSoup(r.text, "html.parser")

        # Meeting Script can be found only in the archive as it is published after five years
        if from_year > 2014:
            print("Meeting scripts are available for 2014 or older")
        if from_year <= 2014:
            for year in range(from_year, 2015):
                yearly_contents = []
                fomc_yearly_url = (
                    self.base_url
                    + "/monetarypolicy/fomchistorical"
                    + str(year)
                    + ".htm"
                )
                r_year = requests.get(fomc_yearly_url)
                soup_yearly = BeautifulSoup(r_year.text, "html.parser")
                meeting_scripts = soup_yearly.find_all(
                    "a", href=re.compile("^/monetarypolicy/files/FOMC\d{8}meeting.pdf")
                )
                for meeting_script in meeting_scripts:
                    self.links.append(meeting_script.attrs["href"])
                    self.speakers.append(
                        self._speaker_from_date(
                            self._date_from_link(meeting_script.attrs["href"])
                        )
                    )
                    self.titles.append("FOMC Meeting Transcript")
                    self.dates.append(
                        datetime.strptime(
                            self._date_from_link(meeting_script.attrs["href"]),
                            "%Y-%m-%d",
                        )
                    )
                if self.verbose:
                    print(
                        "YEAR: {} - {} meeting scripts found.".format(
                            year, len(meeting_scripts)
                        )
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

        link_url = self.base_url + link
        pdf_filepath = (
            self.output_raw_dir
            + "/FOMC_MeetingScript_"
            + self._date_from_link(link)
            + ".pdf"
        )

        if not os.path.exists(pdf_filepath) or self.force_download:
            # Scripts are provided only in pdf. Save the pdf and pass the content
            res = requests.get(link_url)
            with open(pdf_filepath, "wb") as f:
                f.write(res.content)
        else:
            if self.verbose:
                print("File already exists: ", pdf_filepath)
        # Extract text from the pdf
        pdf_file_parsed = ""  # new line
        with pdfplumber.open(pdf_filepath) as pdf:
            for page in pdf.pages:
                pg_width = page.width
                pg_height = page.height
                pg_bbox = (
                    self.crop_coords[0] * float(pg_width),
                    self.crop_coords[1] * float(pg_height),
                    self.crop_coords[2] * float(pg_width),
                    self.crop_coords[3] * float(pg_height),
                )
                page_crop = page.crop(bbox=pg_bbox)
                text = page_crop.extract_text()
                pdf_file_parsed = pdf_file_parsed + "\n" + text
        paragraphs = re.sub("(\n)(\n)+", "\n", pdf_file_parsed.strip())
        paragraphs = paragraphs.split("\n")

        section = -1
        paragraph_sections = []
        for paragraph in paragraphs:
            if not re.search(
                "^(page|january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)",
                paragraph.lower(),
            ):
                if len(re.findall(r"[A-Z]", paragraph[:10])) > 5 and not re.search(
                    "(present|frb/us|abs cdo|libor|rp–ioer|lsaps|cusip|nairu|s cpi|clos, r)",
                    paragraph[:10].lower(),
                ):
                    section += 1
                    paragraph_sections.append("")
                if section >= 0:
                    paragraph_sections[section] += paragraph
        self.articles[index] = self.segment_separator.join(
            [paragraph for paragraph in paragraph_sections]
        )
