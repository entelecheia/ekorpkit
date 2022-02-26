# Copyright (c) 2019 NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from bs4 import BeautifulSoup
import wget


class PubMed:
    def __init__(self, subset, output_dir, force_download=False, **kwargs):
        self.subset = subset
        self.force_download = force_download

        self.output_dir = output_dir + "/" + subset
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.download_urls = {
            "baseline": "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/",
            "daily_update": "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/",
            "oa_comm": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_comm/xml/",
            "oa_noncomm": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/",
            "oa_other": "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_other/xml/",
        }
        self.download()

    def download(self):
        print("subset:", self.subset)
        if os.listdir(self.output_dir) and not self.force_download:
            print("Files already downloaded. Skipping.")
            return
        url = self.download_urls[self.subset]
        self.download_files(url)

    def download_files(self, url):
        url = self.download_urls[self.subset]
        output = os.popen("curl " + url).read()
        soup = BeautifulSoup(output, "html.parser")
        links = soup.find_all("a")
        filelist = [link.attrs["href"] for link in links]

        if self.subset in ["oa_comm", "oa_noncomm", "oa_other"]:
            filelist = [file for file in filelist if file.endswith(".tar.gz")]
        elif self.subset == "baseline" or self.subset == "daily_update":
            filelist = [file for file in filelist if file.endswith(".gz")]
        else:
            assert False, "Invalid PubMed dataset/subset specified."

        print(f"Number of files: {len(filelist)}")
        for i, filename in enumerate(filelist):
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.isfile(filepath) or os.path.getsize(filepath) == 0:
                print(f"\nDownloading file {i+1} of {len(filelist)}: {filename}")
                wget.download(url + filename, filepath)
            else:
                print(f"{filename} already downloaded. Skipping.")
