# NIKL Wrtten Text Corpus
 
[Sample](../sample/nikl_written.txt)
 
<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=../../../ekorpkit/resources/corpora/nikl_written.yaml) -->
<!-- The below code snippet is automatically added from ../../../ekorpkit/resources/corpora/nikl_written.yaml -->
```yaml
name: nikl_written
fullname: NIKL Wrtten Text Corpus
lang: ko
category: formal
description: NIKL Wrtten Text Corpus
license: National Institute of the Korean Language Corpus - Wrtten
homepage: https://corpus.korean.go.kr
version: 1.0.0
num_docs: 20128
num_docs_before_processing: 20188
num_segments: 20159
num_sents: 27231846
num_words: 679547033
size_in_bytes: 6922971630
num_bytes_before_processing: 6962611396
size_in_human_bytes: 6.45 GiB
data_files_modified: '2022-02-25 01:35:31'
meta_files_modified: '2022-02-22 11:44:34'
info_updated: '2022-02-26 03:06:08'
data_files:
  train: nikl_written-train.parquet
meta_files:
  train: meta-nikl_written-train.parquet
column_info:
  keys:
    id: id
    text: text
  data:
    id: int
    text: str
  meta:
    id: int
    doc_id: str
    title: str
    author: str
    publisher: str
    date: str
    category: str
```
<!-- MARKDOWN-AUTO-DOCS:END -->
