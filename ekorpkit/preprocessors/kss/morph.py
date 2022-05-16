#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Korean Sentence Splitter
# Split Korean text into sentences using heuristic algorithm.
#
# Copyright (C) 2021 Hyun-Woong Ko <kevin.ko@tunib.ai> and Sang-Kil Park <skpark1224@hyundai.com>
# All rights reserved.
#
# This software may be modified and distributed under the terms
# of the BSD license.  See the LICENSE file for details.

from .base import Eojeol


class MorphExtractor(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.parse_args(**kwargs)

    def parse_args(self, **kwargs):
        self.kwargs = kwargs
        self.backend = kwargs.get("backend", "mecab-python3")
        self.dicdir = kwargs.get("dicdir", None)
        self.userdic_path = kwargs.get("userdic_path", None)

    def pos(self, text):
        if self.backend in ["mecab-python3", "fugashi"]:
            from ekorpkit.models.tokenizer.mecab import MeCab

            try:
                self.mecab = MeCab(
                    dicdir=self.dicdir,
                    userdic_path=self.userdic_path,
                    backend=self.backend,
                )
            except ImportError:
                raise ImportError(
                    "\n"
                    "You must install [`fugashi` or `mecab-python3`] and `mecab_ko_dic` if you want to use `fugashi` backend.\n"
                )
        else:
            raise AttributeError(
                "Wrong backend ! currently, we only support `fugashi`, `mecab-python3`, `none` backend."
            )
        return [
            Eojeol(eojeol, pos[1]) for pos in self.mecab.pos(text) for eojeol in pos[0]
        ]
