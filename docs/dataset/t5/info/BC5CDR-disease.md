# BC5CDR-disease
 
<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=../../../../ekorpkit/resources/datasets/t5/BC5CDR-disease.yaml) -->
<!-- The below code snippet is automatically added from ../../../../ekorpkit/resources/datasets/t5/BC5CDR-disease.yaml -->
```yaml
name: BC5CDR-disease
domain: biomed
task: ner
lang: en
num_examples: 13938
size_in_bytes: 2043073
size_in_human_bytes: 1.95 MiB
data_files_modified: '2022-02-26 03:02:29'
info_updated: '2022-02-26 03:06:01'
data_files:
  train: BC5CDR-disease-train.csv
  dev: BC5CDR-disease-dev.csv
  test: BC5CDR-disease-test.csv
column_info:
  columns:
    id: id
    text: input_text
  data:
    id: int
    prefix: str
    input_text: str
    target_text: str
```
<!-- MARKDOWN-AUTO-DOCS:END -->
