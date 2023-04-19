# BC4CHEMD
 
<!-- MARKDOWN-AUTO-DOCS:START (CODE:src=../../../../ekorpkit/resources/datasets/t5/BC4CHEMD.yaml) -->
<!-- The below code snippet is automatically added from ../../../../ekorpkit/resources/datasets/t5/BC4CHEMD.yaml -->
```yaml
name: BC4CHEMD
domain: biomed
task: ner
lang: en
num_examples: 87685
size_in_bytes: 14480974
size_in_human_bytes: 13.81 MiB
data_files_modified: '2022-02-26 03:02:00'
info_updated: '2022-02-26 03:06:01'
data_files:
  train: BC4CHEMD-train.csv
  dev: BC4CHEMD-dev.csv
  test: BC4CHEMD-test.csv
features:
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
