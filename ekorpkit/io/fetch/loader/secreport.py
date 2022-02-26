import os
import pandas as pd

# from pandarallel import pandarallel
from datetime import datetime, timedelta
import re

from ekorpkit.io.db import connect_mongo
from ekorpkit.preprocessors.normalizer import repeat_normalize


class SecReport:
    REMOVE_PATTERNS_1 = [
        # 한글, 영어, 띄어쓰기, 일부 특수 문자 등을 제외하고 모두 제거
        r"[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+",
        # 아래 두개는 각주 형식이라서 있으면 이상하게 문자열 끝에 추가됨;;
        r"\[\*[^\]]+\]",
        r"~~[^~]+~~",
    ]
    REMOVE_PATTERNS_2 = [
        # EMAIL_PATTERN
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        # URL_PATTERN
        r"(ftp|http|https)?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        # HTML tags
        r"<[a-zA-Z]*?>",
        # 빈 괄호 제거
        r"(\(|\[|<)+[^가-힣A-Za-z\d]*(>|\]|\))+",
        # (, xxxxx) 에서 , 제거
        r"(?<=[(\(|\[|<)+])\s*(\s*,\s*)+\s*",
        # [*** Not working ***] (xxxxx , ) 에서 , 제거
        # r'\s*(\s*,\s*)+\s*(?=[(>|\]|\))+]+)'
        # 괄호와 그 내용들 제거, 안녕(하세요) -> 안녕
        # r"\([^\)]*\)"
    ]

    REPLACE_PATTERNS = {
        # \n -> 띄어쓰기
        # '\\n': "\n",
        # \' -> '
        # "\\'": "'",
        # 빈 괄호 제거
        r"(\(|\[|<)+\s*(>|\]|\))+": "",
        # MULTIPLE_SPACES
        r" +": " ",
        # WIKI_SPACE_CHARS
        r"(\\s|゙|゚|　)+": " ",
        # MULTIPLE_NEWLINES
        r"(?<!\s)(?<!\.)\s*\n\s+\n": "\n",
        r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=[다]\.|[다까]\?|[다]\!|[가-힣A-Za-z\s]=)\s+": "\n"
        # r'\s*\n\s*\n\s*': '  \t'
    }

    SKIP_SENTENCE_PATTERNS = {
        # WIKI_REMOVE_CHARS
        r"'+|(=+.{2,30}=+)|__TOC__|(ファイル:).+|:(en|de|it|fr|es|kr|zh|no|fi):",
        # [글 = 정진건 기자], ???  < 논설위원 겸 경제교육연구소장 >
        # r'(\[|\(|<|)\s?(글\s?=\s?)?([가-힣A-Za-z·, ]{2,6}\s*){1,2}(기자|특파원),?\s*(\]|\)|>|.?=|)(?![가-힣])'
        r"\[[^\]]*(저작권자|ⓒ|=|기자|특파원)[^\]]*\]",
        r"\([^\)]*(저작권자|ⓒ|=|기자|특파원)[^\)]*\)",
        r"<[^>]*(저작권자|ⓒ|=|기자|특파원)[^>]*>",
        r"(본|이).?기사는.+기사.*입니다",
        r"무단전재.*금지",
        # Wiki section
        r"Section:+",
    }

    def __init__(
        self,
        name,
        output_dir,
        output_file,
        limit=0,
        mongo=None,
        debug_mode=False,
        num_workers=1,
        normalize=None,
        force_download=False,
        **kwargs,
    ):
        self.name = name
        self.reports_df = None
        self.output_path = f"{output_dir}/{output_file}"
        os.makedirs(output_dir, exist_ok=True)
        self.limit = limit
        self.num_workers = num_workers
        self.debug_mode = debug_mode
        self.min_len = normalize.min_len
        self.min_words = normalize.min_words
        self.num_repeats = normalize.num_repeats

        if os.path.exists(self.output_path) and not force_download:
            print(f"output file {self.output_path} already exists, skipping")
        else:
            self.mongo = connect_mongo(**mongo)
            self.results = self._iter_reports()
            self.extract_reports()

    def extract_reports(self):
        reports_text = []
        for rst in self.results:
            # print(rst)
            doc = self._preprocess_report(rst)
            # print(doc)]
            if doc:
                reports_text.append(doc)
        # Store the output in compressed format
        self.reports_df = pd.DataFrame(reports_text)
        self.normalize()
        self.reports_df.to_csv(self.output_path, header=True)
        print(self.reports_df.tail())
        print(f"Corpus [{self.name}] is built to [{self.output_path}]")

    def normalize(self):
        text_key = "text"

        def normalize(row):
            text = row[text_key]
            return self._normalize_text(text)

        # if self.num_workers > 1:
        # 	pandarallel.initialize(nb_workers=self.num_workers)
        # elif self.num_workers < 1:
        # 	pandarallel.initialize()

        df = self.reports_df
        # if self.num_workers == 1:
        df[text_key] = df.apply(normalize, axis=1)
        # else:
        # 	df[text_key] = df.parallel_apply(normalize, axis=1)

        self.reports_df = df.dropna()
        print("Normalization complete!")

    def _normalize_text(self, text):
        # remove non-UTF
        text = text.encode("utf-8", "ignore").decode()

        for pattern in self.REMOVE_PATTERNS_1:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            text = re.sub(regex, "", text)
        for pattern in self.REMOVE_PATTERNS_2:
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            text = re.sub(regex, "", text)
        for pattern, v in self.REPLACE_PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            text = re.sub(regex, v, text)

        paragraphs = []
        for paragraph in re.split(r"\n\n", text):
            sentences = []
            for s in re.split(r"\n", paragraph):
                skip = False
                for pattern in self.SKIP_SENTENCE_PATTERNS:
                    regex = re.compile(pattern, re.IGNORECASE)
                    pos = re.search(regex, s)
                    if pos:
                        skip = True
                        break

                if (
                    len(s) >= self.min_len
                    and len(s.split()) >= self.min_words
                    and not skip
                ):
                    if self.num_repeats > 1:
                        s = repeat_normalize(s, num_repeats=self.num_repeats)
                    sentences.append(s.strip())
            if len(sentences) > 0:
                paragraphs.append(sentences)
        if len(paragraphs) > 0:
            text = "\n\n".join(["\n".join(p) for p in paragraphs])
        else:
            text = None
        return text

    def _iter_reports(self):
        for item in self.mongo.find({"DT": {"$exists": "true"}}).limit(self.limit):
            rid = item["_id"]
            rdate = datetime.strptime(item["DT"], "%Y%m%d") + timedelta(hours=8)
            body = item["BODY"]
            rpt_typ = item["RPT_TYP"]
            rpt_sub_typ = item["RPT_SUB_TYP"]
            sec_cd = item["SEC_CD"]
            yield rid, rdate, body, rpt_typ, rpt_sub_typ, sec_cd

    def _preprocess_report(self, rst):
        rid, rdate, body, rpt_typ, rpt_sub_typ, sec_cd = rst

        if self.debug_mode:
            print(rid, rdate, len(body))
        paragraphs = []
        for p, paragraph in enumerate(body):
            sentences = []
            # print('-----------------')
            for i, sentence in enumerate(paragraph):
                sentence = re.sub(" +", " ", sentence).strip()

                pattern = "[가-힣A-Za-z\d'\"]"
                pos = re.search(pattern, sentence)
                if pos:
                    sentence = sentence[pos.start() :].strip()
                    if len(sentence) > 0:
                        pattern = "(당사는.*(유동성\s*공급자|위탁\s*증권사|동\s*자료)|^(본|이)\s*(조사자료|보고서|리포트)(는|의))"
                        pos = re.search(pattern, sentence)
                        if pos:
                            if self.debug_mode:
                                print(sentence)
                                print("----------------------------skipping sentences")
                            break
                        if sentence not in sentences:
                            sentences.append(sentence)
            if len(sentences) > 0:
                paragraphs.append(sentences)

        rid = "r" + str(rid)

        if len(paragraphs) < 1:
            if self.debug_mode:
                print(" ### skipping a report with not enough number of sentences...")
            return None

        nos = sum([len(p) for p in paragraphs])
        if nos < 4:
            if self.debug_mode:
                print(" ### skipping a report with not enough number of sentences...")
            return None
        # if debug:
        #     print(paragraphs)
        text = "\n\n".join(["\n".join(p) for p in paragraphs])
        if self.debug_mode:
            print(text)

        doc = {
            "rid": rid,
            "rdate": rdate,
            "type": rpt_typ,
            "subtype": rpt_sub_typ,
            "nos": nos,
            "text": text,
        }
        return doc
