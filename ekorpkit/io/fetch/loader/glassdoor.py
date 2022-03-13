import tarfile
import orjson as json
from pathlib import Path
from bs4 import BeautifulSoup
import multiprocessing as mp


class GlassDoor:
    def __init__(
        self,
        name,
        input_file,
        output_dir,
        output_file,
        file_extension,
        num_workers,
        force_download,
        **kwargs,
    ):

        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_file

        if output_path.is_file() and not force_download:
            print(f"{output_path} already exists, skipping")
            return

        with tarfile.open(input_file) as tar:
            files = [file for file in tar.getmembers() if file_extension in file.name]
            print(f"Number of files in [{input_file}]: {len(files)}")
            processes = num_workers if num_workers else 1
            pool = mp.Pool(processes=processes)
            results = []
            for fno, file in enumerate(files):
                print("==> processing {}th file: {}".format(fno + 1, file.name))
                f = tar.extractfile(file)
                if f:
                    data = f.read()
                    results.append(pool.apply_async(parse_gd_review_html, args=(data,)))
            pool.close()
            pool.join()
            with open(output_path, "w") as f:
                for result in results:
                    reviews = result.get()
                    for line in reviews:
                        json.dump(line, f)
                        f.write("\n")

        print(f"Corpus [{name}] is built to [{output_path}] from [{input_file}]")


def parse_gd_review_html(data):
    soup = BeautifulSoup(data, "html.parser")
    reviews = []
    rvs = soup.find_all("li")
    for i, rv in enumerate(rvs):
        if rv.get("id") and rv.get("id").startswith("empReview_"):
            # print(f'parsing review {i}')
            review_dict = parse_gd_review(rv)
            reviews.append(review_dict)
            # print(review_dict)
    return reviews


def parse_gd_review(review):
    review_dict = {}

    rv_id = review.get("id")
    review_dict["id"] = rv_id
    rv_date = review.find("time", class_="date subtle small")
    rv_date = rv_date.get("datetime") if rv_date else None
    review_dict["date"] = rv_date
    rv_title = review.find("h2", class_="h2 summary strong mt-0 mb-xsm")
    if rv_title:
        rv_link = rv_title.find("a")
        if rv_link:
            rv_link = rv_title.find("a").get("href")
        rv_title = rv_title.get_text().replace('"', "")
    review_dict["link"] = rv_link
    review_dict["title"] = rv_title

    author = review.find("span", class_="authorJobTitle middle reviewer")
    author = author.get_text() if author else None
    review_dict["author"] = author

    ratings = {}
    srating = review.find(
        "div", class_="subRatings module stars__StarsStyles__subRatings"
    )
    if srating:
        for li in srating.find_all("li"):
            cat = li.find("div").get_text()
            rating = li.find("span", class_="gdBars gdRatings med").get("title")
            ratings[cat] = rating

    review_dict["ratings"] = ratings

    reviews = {}
    review_txt = ""
    rvtexts = review.find_all(
        "div", class_="mt-md common__EiReviewTextStyles__allowLineBreaks"
    )
    if rvtexts:
        for rvtext in rvtexts:
            text = ""
            cat = ""
            for p in rvtext.find_all("p"):
                if p.get("class"):
                    if p.get("class") == ["strong"]:
                        cat = p.get_text()
                else:
                    text += p.get_text().replace("\r", " ").strip() + "\n"
            if cat:
                reviews[cat] = text.strip()
            review_txt += text + "\n"
    review_dict["reviews"] = reviews
    review_dict["text"] = review_txt.strip()

    return review_dict
