import logging
import rubrix as rb
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class Rubrix:
    def __init__(self, **args):
        args = eKonf.to_config(args)
        self.args = args
        self.name = args.name
        self.verbose = args.get("verbose", False)
        self._api_key = args.get("api_key") or eKonf.osenv("RUBRIX_API_KEY")
        self._api_url = args.get("api_url") or eKonf.osenv("RUBRIX_API_URL")
        self._workspace = args.get("workspace") or eKonf.osenv("RUBRIX_WORKSPACE")
        self.auto = args.auto

        if self.auto.init:
            self.init_rubrix()

    def init_rubrix(self):
        log.info(f"Initializing Rubrix: {self._api_url}, workspace: {self._workspace}")
        rb.init(api_url=self._api_url, api_key=self._api_key, workspace=self._workspace)

    def set_workspace(self, workspace):
        rb.set_workspace(workspace)
        self._workspace = workspace

    def get_workspace(self):
        return rb.get_workspace()

    def create_records_from_cv_preds(
        self,
        cv_preds,
        columns=None,
        meta_columns=["id", "split"],
        prediction_agent=None,
    ):
        columns = columns or self.args.columns
        text_col = columns["text"]
        label_col = columns["labels"]
        pred_col = columns["model_outputs"]
        records = []
        for i, row in cv_preds.iterrows():
            record = rb.TextClassificationRecord(
                inputs={"text": row[text_col]},
                prediction=list(row[pred_col].items()),
                annotation=row[label_col] if label_col in cv_preds.columns else None,
                metadata=row[meta_columns].to_dict(),
                prediction_agent=prediction_agent,
            )
            records.append(record)
        log.info(f"Created {len(records)} records")
        return records

    def find_label_errors(self, records):
        from rubrix.labeling.text_classification import find_label_errors

        label_errors = find_label_errors(records)
        log.info(f"Found {len(label_errors)} label errors")
        return label_errors

    def log(self, records, name):
        rb.log(records, name=name)
        log.info(f"Logged {len(records)} records")

    def delete(self, name):
        rb.delete(name)
        log.info(f"Deleted {name}")

    def copy(self, dataset, name_of_copy):
        rb.copy(dataset, name_of_copy)
        log.info(f"Copied {dataset} to {name_of_copy}")

    def rename(self, dataset, new_name):
        rb.copy(dataset, new_name)
        rb.delete(dataset)
        log.info(f"Renamed {dataset} to {new_name}")

    def load(
        self, name, query=None, ids=None, limit=None, id_from=None, as_pandas=True
    ):
        dset = rb.load(name, query=query, ids=ids, limit=limit, id_from=id_from)
        if as_pandas:
            dset = dset.to_pandas()
        return dset

    def update_label_errors(self, original_data, correct_data, id="id", split="train"):
        log.info(f"Updating {len(correct_data)} records")
        if "annotation_agent" not in original_data.columns:
            original_data["annotation_agent"] = None
        original_data["original_labels"] = original_data["labels"]
        for i, row in correct_data.iterrows():
            correct_labels = row["annotation"]
            agent = row["annotation_agent"]
            metadata = row["metadata"]
            status = row["status"]
            if split and metadata["split"] != split:
                continue
            if status == "Validated":
                original_data.loc[
                    (original_data[id] == metadata[id])
                    & (original_data["split"] == split),
                    ["labels", "annotation_agent", "status"],
                ] = [correct_labels, agent, status]
            elif status == "Discarded":
                original_data = original_data[
                    ~(
                        (original_data[id] == metadata[id])
                        & (original_data["split"] == split)
                    )
                ]
        return original_data

    def remove_label_errors(self, original_data, error_records, id="id", split="train"):
        log.info(f"Removing {len(error_records)} records")
        for record in error_records:
            metadata = record.metadata
            if split and metadata["split"] != split:
                continue
            original_data = original_data[
                ~(
                    (original_data[id] == metadata[id])
                    & (original_data["split"] == split)
                )
            ]

        return original_data
