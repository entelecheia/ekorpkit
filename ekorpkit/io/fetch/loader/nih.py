import requests
import wget
import logging
import os
from bs4 import BeautifulSoup


log = logging.getLogger(__name__)


class ExPORTER:
    def __init__(self, output_dir, force_download=False, **kwargs):
        self.force_download = force_download
        self.output_dir = output_dir

        self.download_urls = {
            "exporter": "https://exporter.nih.gov/ExPORTER_Catalog.aspx?sid=0&index=1",
            "crisp": "https://exporter.nih.gov/CRISP_Catalog.aspx?sid=0&index=1",
        }
        self.targets = {
            "exporter": "CSVs/final/RePORTER_",
            "crisp": "CRISP/Abstracts/CRISP_PRJABS_C_",
        }
        self.base_url = "https://exporter.nih.gov/"
        self.download()

    def download(self):
        for subset in self.targets.keys():
            self.download_files(subset)

    def download_files(self, subset):
        log.info(f"\nsubset: {subset}\n")
        output_dir = self.output_dir + "/" + subset
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        if os.listdir(output_dir) and not self.force_download:
            log.info("Files already downloaded. Skipping.")
            return

        url = self.download_urls[subset]
        target = self.targets[subset]
        self.sess = requests.session()
        r = self.sess.get(url)
        soup = BeautifulSoup(r.content, "lxml")
        links = [a["href"] for a in soup.find_all("a", href=True)]
        links = [link for link in links if target in link]

        log.info(f"Found {len(links)} links in Exporter")
        for i, link in enumerate(links):
            filename = os.path.basename(link)
            filepath = os.path.join(output_dir, filename)
            if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
                log.info(f"\nDownloading file {i+1} of {len(links)}: {link}")
                wget.download(self.base_url + link, filepath)
            else:
                log.info(f"File {filename} already exists. Skipping.")
