import logging
import os
import pandas as pd
from abc import ABCMeta, abstractmethod
from ekorpkit import eKonf


log = logging.getLogger(__name__)


class BaseFetcher:
    __metaclass__ = ABCMeta

    def __init__(self, **args):
        self.args = eKonf.to_config(args)
        self.name = self.args.name
        self.auto = self.args.auto
        self.force = self.args.force
        self.verbose = self.args.get("verbose", True)

        self._path = self.args.path
        self.output_dir = self._path.output.base_dir
        self.output_file = self._path.output.filepath

    def fetch(self):
        if not self._exsits() or self.force.download:
            self._fetch()
        else:
            log.info(f"{self.output_file} already exists. skipping..")

    def _exsits(self):
        return os.path.exists(self.output_file)

    def _fetch(self):
        raise NotImplementedError
