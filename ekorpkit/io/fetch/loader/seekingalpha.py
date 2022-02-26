import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from ekorpkit.utils.batch.batcher import tqdm_joblib


class EarningsCall:
    def __init__(
        self,
        name,
        input_file,
        output_dir,
        output_file,
        filetype,
        force_download,
        num_workers,
        **kwargs,
    ):
        import tarfile

        output_dir = Path(output_dir)
        if not output_dir.is_dir():
            output_dir.mkdir(exist_ok=True)
        output_path = output_dir / output_file

        if output_path.exists() and not force_download:
            print(f"Corpus [{name}] is already built to [{output_path}]")
            return

        documents = []
        with tarfile.open(input_file) as tar:
            files = [file for file in tar.getmembers() if filetype in file.name]
            print(f"Number of files in [{input_file}]: {len(files)}")
            processes = num_workers if num_workers else 1
            desciption = f"Extracting {filetype} files"
            with tqdm_joblib(tqdm(desc=desciption, total=len(files))) as pbar:
                results = Parallel(n_jobs=processes)(
                    delayed(html_to_document)(tar.extractfile(file).read(), file.name)
                    for file in files
                )
                for result in results:
                    documents.append(result)

            # pool = mp.Pool(processes=processes)
            # results = []
            # for file in tqdm(files):
            # 	data = tar.extractfile(file).read()
            # 	results.append(pool.apply_async(html_to_document, args=(data, file.name)))
            # pool.close()
            # pool.join()
            # for result in results:
            # 	doc = result.get()
            # 	documents.append(doc)

        reports_df = pd.DataFrame(documents)
        reports_df.to_csv(output_path, header=True)
        print(reports_df.tail())
        print(f"Corpus [{name}] is built to [{output_path}] from [{input_file}]")


def html_to_document(data, filename):
    soup = BeautifulSoup(data, "html.parser")
    ps = soup.find_all("p")
    text = ""
    title = ""
    for i, p in enumerate(ps):
        if i == 0:
            title = p.text.strip()
        else:
            text += p.text + "\n"
    doc = {
        "filename": filename,
        "title": title,
        "nos": len(text.split("\n")),
        "text": text,
    }
    return doc
