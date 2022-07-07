import logging
import os
import re
import glob
import requests
import subprocess
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from ekorpkit.io.fetch.base import BaseFetcher
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BOKMniutes(BaseFetcher):
    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        super().__init__(**args)
        self.from_date = str(self.args.scrap.from_date)
        self.base_url = self.args.scrap.base_url
        self.url = self.args.scrap.url
        self.raw_hwp_dir = self.args.scrap.raw_hwp_dir
        self.raw_txt_dir = self.args.scrap.raw_txt_dir
        os.makedirs(self.raw_hwp_dir, exist_ok=True)
        os.makedirs(self.raw_txt_dir, exist_ok=True)
        self.file_prefix = self.args.scrap.file_prefix

        user_agent = "Mozilla/5.0"
        self.headers = {"User-Agent": user_agent}
        self.minutes_urls = []
        self.minutes_hwp_files = []
        self.minutes_txt_files = []

        if self.auto.load:
            self.fetch()

    def _fetch(self):
        self.get_minutes_list()
        self.get_minutes_files()
        self.convert_hwp_to_txt()
        self.build_minutes()

    def build_minutes(self):

        docs = []
        for filepath in glob.glob(os.path.join(self.raw_txt_dir, "*.txt")):
            doc = preprocess_minutes(filepath, self.file_prefix)
            docs += doc

        data = pd.DataFrame(
            docs, columns=["id", "filename", "mdate", "rdate", "section", "text"]
        )
        data = data.dropna()

        if not self.force.download:
            if os.path.isfile(self.output_file):
                log.info(
                    "minutes file already exists. combining with the existing file.."
                )
                minutes_df = pd.read_csv(self.output_file, index_col=None)
                data = minutes_df.combine_first(data)
                data = data.drop_duplicates(subset=["id"])

        eKonf.save_data(data, self.output_file, verbose=self.verbose)
        if self.verbose:
            print(data.tail())
        log.info(f"Saved {len(docs)} documents to {self.output_file}")

    def convert_hwp_to_txt(self):
        for filepath in glob.glob(os.path.join(self.raw_hwp_dir, "*.hwp")):
            filename = os.path.basename(filepath)
            hwp_file = f"{self.raw_hwp_dir}/{filename}"
            txt_file = f"{self.raw_txt_dir}/{filename[:-4]}.txt"
            self.minutes_txt_files.append(txt_file)

            if os.path.exists(txt_file):
                log.info(f"{txt_file} already exists")
                continue

            command = f"hwp5txt --output={txt_file} {hwp_file}"
            subprocess.Popen(command, shell=True)
            log.info(f"converted {hwp_file} to {txt_file}")

    def get_minutes_files(self):
        for info in self.minutes_urls:
            self.get_minutes_file(
                info["url"], info["meeting_date"], info["release_date"]
            )

    def get_minutes_file(self, page_addr, mdate, rdate):
        filename = (
            self.file_prefix
            + "_"
            + mdate.strftime("%Y%m%d")
            + "_"
            + rdate.strftime("%Y%m%d")
            + ".hwp"
        )
        filepath = f"{self.raw_hwp_dir}/{filename}"

        if os.path.exists(filepath):
            self.minutes_hwp_files.append(filepath)
            print(f"{filepath} already exists")
            return

        page = requests.get(page_addr)
        soup = BeautifulSoup(page.content, "html.parser")
        links = soup.find("div", class_="addfile").find_all("a")
        for link in links:
            link_filename = link.get_text()
            link_filename = (
                link_filename.replace("\r", "").replace("\t", "").replace("\n", "")
            )
            if link_filename[-3:] == "hwp":
                file_addr = self.base_url + link["href"]
                file_res = requests.get(file_addr)

                with open(filepath, "wb") as f:
                    f.write(file_res.content)
                self.minutes_hwp_files.append(filepath)
                print(f"saved {link_filename} as {filename} from {file_addr}")

    # Because this is for the recent minutes, changed the address of the list to that of the rss feed of the BOK minutes.
    def get_minutes_list(self):
        from_date = datetime.strptime(self.from_date, "%Y%m%d")
        url = self.url
        page = requests.get(url, headers=self.headers)

        soup = BeautifulSoup(page.content, "html.parser")
        brdList = soup.find_all("item")

        for post in brdList:
            pubdate = post.find("pubdate").get_text().strip()
            guid = post.find("guid").get_text().strip()
            title = post.find("title").get_text().strip()
            # description = post.find("description").get_text().strip()
            # if description.replace(' ','').find('통화정책방향') >= 0:
            mdate = title[title.find(")(") + 2 : -1]
            if mdate[-1] == ".":
                mdate = mdate[:-1]
            mdate = datetime.strptime(mdate, "%Y.%m.%d")
            if mdate < from_date:
                break

            rdate = pubdate[5:16]
            rdate = datetime.strptime(rdate, "%d %b %Y")
            info = {
                "meeting_date": mdate,
                "release_date": rdate,
                "url": guid,
                "title": title,
            }
            self.minutes_urls.append(info)


def tidy_sentences(section):
    sentence_enders = re.compile(r"((?<=[함음됨임봄짐움])(\s*\n|\.|;)|(?<=다)\.)\s*")
    splits = list((m.start(), m.end()) for m in re.finditer(sentence_enders, section))
    starts = [0] + [i[1] for i in splits]
    ends = [i[0] for i in splits]
    sentences = [section[start:end] for start, end in zip(starts[:-1], ends)]
    for i, s in enumerate(sentences):
        sentences[i] = (s.replace("\n", " ").replace(" ", " ")) + "."

    text = "\n".join(sentences) if len(sentences) > 0 else ""
    return sentences, text


def preprocess_minutes(filepath, file_prefix):
    filename = os.path.basename(filepath).replace(".txt", "")
    fileinfo = filename.split("_")
    mdate = datetime.strptime(fileinfo[1], "%Y%m%d") + timedelta(hours=10)
    rdate = datetime.strptime(fileinfo[2], "%Y%m%d") + timedelta(hours=16)
    fileinfo[0] = file_prefix
    filename = "_".join(fileinfo)
    doc = []

    print("open file: {}".format(filepath))
    minutes = open(filepath, encoding="utf-8").read()

    pos = re.search(
        "(.?국내외\s?경제\s?동향.?과 관련하여,?|\(가\).+경제전망.*|\(가\) 국내외 경제동향 및 평가)\n?\s*일부 위원은",
        minutes,
        re.MULTILINE,
    )
    s1 = pos.start() if pos else -1
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여,?|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    # pos = re.search('(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*일부 위원은', minutes, re.MULTILINE)
    pos = re.search(
        "(.?외환.?국제금융\s?동향.?과 관련하여.*|\(나\) 외환.국제금융\s?(및 금융시장)?\s?동향)\n?\s*(일부 위원은|대부분의 위원들은)",
        minutes,
        re.MULTILINE,
    )
    s2 = pos.start() if pos else -1
    pos = re.search(
        "(.?금융시장\s?동향.?과 관련하여,?|\(다\) 금융시장\s?동향)\n?\s*일부 위원은", minutes, re.MULTILINE
    )
    s3 = pos.start() if pos else -1
    # pos = re.search('((\((다|라)\) )?.?통화정책방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에 관해 다음과 같은 토론이 있었음.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책.?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    # pos = re.search('((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?', minutes, re.MULTILINE)
    pos = re.search(
        "((\((다|라)\) )?.?통화정책\s?방향.?에 관한 토론,?|이상과 같은 의견\s?교환을 바탕으로.*통화정책\s?방향.*에.*토론.*)\n?",
        minutes,
        re.MULTILINE,
    )
    s4 = pos.start() if pos else -1
    pos = re.search("(\(4\) 정부측 열석자 발언.*)\n?", minutes, re.MULTILINE)
    s5 = pos.start() if pos else -1
    pos = re.search(
        "(\(.*\) 한국은행 기준금리 결정에 관한 위원별 의견\s?개진|이상과 같은 토론에 이어 .* 관한 위원별 의견개진이 있었음.*)\n?",
        minutes,
        re.MULTILINE,
    )
    s6 = pos.start() if pos else -1
    # pos = re.search('(\(\s?.*\s?\) ()(심의결과|토의결론))\n?', minutes, re.MULTILINE)
    # s7 = pos.start() if pos else -1
    positer = re.finditer("(\(\s?.*\s?\) ()(심의결과|토의결론))\n?", minutes, re.MULTILINE)
    s7 = [pos.start() for pos in positer if pos.start() > s6]
    s7 = s7[0] if s7 else -1

    # 국내외 경제동향
    bos = s1
    eos = s2
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("(일부|대부분의) 위원들?은", section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section1, section1_txt = tidy_sentences(section)

    # 외환․국제금융 동향
    bos = s2
    eos = s3 if s3 >= 0 else s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("(일부|대부분의) 위원들?은", section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section2, section2_txt = tidy_sentences(section)

    # 금융시장 동향
    bos = s3
    eos = s4
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("(일부|대부분의) 위원들?은", section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section3, section3_txt = tidy_sentences(section)

    # 통화정책방향
    bos = s4
    eos = s5 if s5 >= 0 else s6 if s6 >= 0 else s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("(일부|대부분의) 위원들?은", section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section4, section4_txt = tidy_sentences(section)

    # 위원별 의견 개진
    bos = s6
    eos = s7
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("(일부|대부분의) 위원들?은", section, re.MULTILINE)
    bos = pos.start() if pos else -1
    section = section[bos:] if bos >= 0 else section
    section5, section5_txt = tidy_sentences(section)

    # 정부측 열석자 발언
    bos = s5
    eos = s6
    section = minutes[bos:eos] if bos >= 0 or eos >= 0 else ""
    pos = re.search("정부측 열석자 발언", section, re.MULTILINE)
    bos = pos.end() + 1 if pos else -1
    section = section[bos:] if bos >= 0 else section
    section6, section6_txt = tidy_sentences(section)

    sections = [
        "Economic Situation",
        "Foreign Currency",
        "Financial Markets",
        "Monetary Policy",
        "Participants’ Views",
        "Government’s View",
    ]
    # section_texts = (section1, section2, section3, section4, section5, section6)
    section_texts = (
        section1_txt,
        section2_txt,
        section3_txt,
        section4_txt,
        section5_txt,
        section6_txt,
    )

    print(" ==> text processing completed: {}".format(filename))

    if any(section_texts):
        for s, (section, text) in enumerate(zip(sections, section_texts)):
            # for p, text in enumerate(sentences):
            id = f"{filename}_S{s+1}"
            # id = f'{filename}_S{s+1}_P{p+1}'
            # row = (id, filename, mdate, rdate, section, p, text)
            row = (id, filename, mdate, rdate, section, text)
            doc.append(row)
    else:
        print("Empty!")

    return doc
