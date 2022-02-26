## CLI Configuration

To see the available configurations for CLI run the command:

```ekorpkit --help```

<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=./config_help.txt) -->
<!-- The below code snippet is automatically added from ./config_help.txt -->
```txt
== ekorpkit == 

author: entelecheia
version: 0.1.1.1
description: This package provides corpus management tools such as extraction, trasformation, and tokenization.

ekorpkit Command Line Interface for Hydra

== Configuration groups ==
Compose your configuration from those groups (task=export)
about: default
about/app: default
about/constant: default
cmd: about, convert_electra, default, extract_nouns, fetch_lmdata, finetune, info, listup, lmdata_old, lmdata_task, load_dataset, ls, topic, train, transform
corpus: builtins, default, document, lmdata
corpus/lmdata: _build, aida_paper, aihub_book_sum, aihub_koen_sci, aihub_koen_specialty, aihub_koen_ssci, aihub_law_case, aihub_law_kb, aihub_paper_sum, aihub_patent_entire_sum, aihub_patent_section_sum, aihub_specialty_field, aihub_specialty_field2, bigkinds, bigpatent, bok_minutes, c4_realnewslike, cb_speech, cc_news, courtlistener, earnings_call, edgar, enron_mail, enwiki, esg_report, fomc, gd_review, hacker_news, kaist, kcbert, kcc, kowiki, mc4_ko, modu_news, modu_spoken, modu_written, namuwiki, nih_exporter, oscar_ko, pathobook, philpapers, pmc_comm, pmc_noncomm, pubmed, respec, reuters_financial, sec_report, stackexchange, the_pile, us_equities_news, verbcl, youtube_subtitles
dataset: _gdrive-conll, _hugging, _local-conll, _local-csv, _pretok, _split, bc4chemd, bc5cdr, default, ncbi-disease, pathner
dataset/t5: _default, bc4chemd, ncbi-disease
dir: default
env: default
fetch: lmdata, t5
fetch/downloader: _default, _gcs, _gdrive, bok, enwiki, kowiki, namuwiki, pubmed
fetch/downloader/lmdata: bigpatent, earnings_call, edgar, enron, esg_report, fomc, glassdoor, kcbert, kcc, nih_exporter, pathobook, respec, sec_report
fetch/loader: class, csv, hfds, parser, tf_textline, the_pile
fetch/loader/parser: email, json, jsonlines, pubmed
info: lmdata
info/stat: default
info/table: listup
method/len_bytes: default
method/len_segments: default
method/len_sents: default
method/len_words: default
method/len_wospc: default
mode: debug, default
model: default
model/plm/electra: base, large
model/plm/electra/discriminator: base, large
model/plm/electra/generator: base, large
model/plm/pretrained: electra-discriminator
model/topic: default
model/transformer/finetune: classification, default, ner
pipeline: lmdata_pipeline
pipeline/aggregate_columns: aggregate
pipeline/combine_columns: combine
pipeline/drop_duplicates: drop
pipeline/fillna: fillna
pipeline/filter_length: filter
pipeline/normalize: normalize
pipeline/normalize/normalizer: formal_en, formal_ko, informal_ko
pipeline/rename_columns: rename
pipeline/replace_regex: regex
pipeline/replace_whitespace: whitespace
pipeline/reset_index: reset
pipeline/save_samples: samples
pipeline/segment: segment
pipeline/segment/segmenter: kss, pysbd, pysbd_merge_enko
task: default, lmdata_task
task/corpus: default, lmdata
task/corpus/load: default
task/info: lmdata_doc
task/lmdata: build_vocab, default
task/lmdata/sharding: default
task/ls: default
task/subtask: lmdata_subtask
task/subtask/export_to_text: default
task/topic: default, export_corpus_samples, export_samples
task/train: convert_electra, default, finetune
task/transform: default
task/transform/chunk: default
task/transform/filter: default
task/transform/metadata: default
task/transform/normalize: default
task/transform/normalize_lr: default
task/transform/nouns: default
task/transform/split_sents: default, kss, simple
task/transform/stats: default
task/transform/to_csv: default
task/transform/to_txt: default
task/transform/tokenize: bwp, default, mecab, pynori, pynori_userdict
task/transform/tokens: default
task/vocab/extract_nouns: default
tokenizer: bert_wordpiece
tokenizer/spm: t5

== Config ==
This is the config generated for this run.
You can override everything, for example:
ekorpkit +task/export=your_config_name
-------
_target_: ekorpkit.cli.about
app_name: ${about.app.name}
project: ${app_name}
print_config: false
print_resolved_config: true
ignore_warnings: true
num_workers: 1
about:
  app:
    name: ekorpkit
    author: entelecheia
    version: 0.1.1.1
    description: This package provides corpus management tools such as extraction,
      trasformation, and tokenization.
  constant:
    CORPUS_ALL: all
    CORPUS_DF: dataframe
    CORPUS_CSV: csv
    ID_KEYS:
    - id
    TEXT_KEY: text
dir:
  workspace: /workspace
  project: ${.workspace}/projects/${app_name}
  data: ${.workspace}/data
  log: ${.project}/logs
  tmp: ${.data}/tmp
  cache: ${.tmp}/.cache
  archive: ${.data}/archive
  runtime: ${hydra:runtime.cwd}
  model: ${.data}/models
  dataset: ${.data}/datasets
  corpus: ${.dataset}/corpus/ekorpkit
env:
  os:
    MODIN_ENGINE: ${..distributed_framework.backend}
    MODIN_CPUS: ${oc.select:..distributed_framework.num_workers,50}
  distributed_framework:
    backend: ray
    initialize: true
    num_workers: ${oc.select:num_workers,50}
  ray:
    num_cpus: ${oc.select:..distributed_framework.num_workers,50}
  dask:
    n_workers: ${oc.select:..distributed_framework.num_workers,50}
  batcher:
    procs: ${oc.select:..distributed_framework.num_workers,50}
    minibatch_size: 20000
    backend: ${..distributed_framework.backend}
    task_num_cpus: 1
    task_num_gpus: 0
    verbose: 10
default_mode: true
initialize_modin: false
task:
  num_workers: ${oc.select:num_workers,1}
corpus:
  name: null
  segment_separator: \n\n
  sentence_separator: \n

-------
Powered by Hydra (https://hydra.cc)
Use --hydra-help to view Hydra specific help
```
<!-- MARKDOWN-AUTO-DOCS:END -->
