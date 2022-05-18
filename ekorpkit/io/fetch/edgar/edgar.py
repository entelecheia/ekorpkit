import os
import requests
from bs4 import BeautifulSoup
from ekorpkit import eKonf
from ekorpkit.io.fetch.web import web_download, web_download_unzip


class EDGAR:
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        self.base_url = self.args.base_url
        self.url = self.args.url
        self.output_dir = self.args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.force_download = self.args.force_download
        self.name = self.args.name

        self.build()

    def build(self):
        if self.force_download or not os.listdir(self.output_dir):
            self.download_edgar()
        else:
            print(f"{self.name} is already downloaded")

    def download_edgar(self):

        user_agent = "Mozilla/5.0"
        headers = {"User-Agent": user_agent}
        page = requests.get(self.url, headers=headers)

        soup = BeautifulSoup(page.content, "html.parser")
        filelist = soup.find_all("a", class_="filename")

        for file in filelist:
            link = self.base_url + file.get("href")
            file_path = self.output_dir + "/" + file.get_text().strip()
            web_download(link, file_path, self.name, self.force_download)
