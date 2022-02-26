# linnaeus
 
<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=../../../../ekorpkit/resources/datasets/t5/linnaeus.yaml) -->
<!-- The below code snippet is automatically added from ../../../../ekorpkit/resources/datasets/t5/linnaeus.yaml -->
```yaml
name: linnaeus
domain: biomed
task: ner
lang: en
num_examples: 23155
size_in_bytes: 2833111
size_in_human_bytes: 2.70 MiB
data_files_modified: '2022-02-26 03:03:26'
info_updated: '2022-02-26 03:06:01'
data_files:
  train: linnaeus-train.csv
  dev: linnaeus-dev.csv
  test: linnaeus-test.csv
column_info:
  keys:
    id: id
    text: input_text
  data:
    id: int
    prefix: str
    input_text: str
    target_text: str
```
<!-- MARKDOWN-AUTO-DOCS:END -->
