import logging
import os
from ekorpkit import eKonf
from ekorpkit.io.file import load_dataframe, save_dataframe

log = logging.getLogger(__name__)


class PredictionData:
    def __init__(self, **args):
        self.args = eKonf.to_dict(args)
        self.pred_data = {}
        self.to_predict = {}

    def load_pred_data(self):
        data_dir = self.args["data_dir"]
        data_files = self.args["data_files"]
        columns_to_keep = self.args["columns_to_keep"]
        self.pred_keys = self.args["keys"]
        self.input_text_key = self.pred_keys["input_text"]
        self.prediction_key = self.pred_keys["prediction"]

        if data_files is None:
            log.warning("No data files are provided")
            return

        if isinstance(data_files, str):
            data_files = [data_files]
        for data_file in data_files:
            log.info(f"Loading {data_file}")
            filepath = os.path.join(data_dir, data_file)
            df = load_dataframe(filepath, verbose=self.verbose)
            if columns_to_keep is not None:
                df = df[columns_to_keep]
            if self.verbose:
                print(df.tail())
            data_file = os.path.basename(data_file)
            self.pred_data[data_file] = df
            to_predict = df[self.input_text_key].tolist()
            if self.verbose:
                print(to_predict[:5])
            self.to_predict[data_file] = to_predict

    def save_predictions(self):
        for data_file, preds in self.predictions.items():
            log.info(f"Saving predictions for {data_file}")
            df = self.pred_data[data_file]
            df[self.prediction_key] = preds
            filepath = os.path.join(self.args.pred_output_dir, data_file)
            save_dataframe(df, filepath, verbose=self.verbose)
