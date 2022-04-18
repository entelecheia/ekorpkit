import os
import re
import time
import codecs
import pandas as pd
from glob import glob
from ekorpkit import eKonf
from omegaconf.dictconfig import DictConfig
from pytablewriter import MarkdownTableWriter
from pytablewriter.style import Style
from pathlib import Path
from ekorpkit.utils.func import humanbytes, get_modified_time
import ekorpkit


def info_docs(**args):
    cfg = eKonf.to_config(args)

    info_cfg = cfg.info
    table_cfg = info_cfg.table
    info_name = info_cfg.name

    if info_cfg.base_dir is None:
        info_cfg.base_dir = Path(ekorpkit.__path__[0]).parent.as_posix()
    sample_output_dir = info_cfg.get("sample_output_dir", None)
    if sample_output_dir:
        sample_output_dir = f"{info_cfg.base_dir}/{sample_output_dir}"
        os.makedirs(sample_output_dir, exist_ok=True)

    info_paths = glob(info_cfg.info_files)
    markdown_template = codecs.decode(info_cfg.markdown_template, "unicode_escape")
    markdown_template = re.sub(r"\s[3,]", "\n\n", markdown_template)
    readme_md_template = codecs.decode(info_cfg.readme_md_template, "unicode_escape")
    readme_md_template = re.sub(r"\s[3,]", "\n\n", readme_md_template)

    infos = []
    for info_path in info_paths:
        print(f"Reading {info_path}")
        info = eKonf.load(info_path)

        data_file_dir = Path(info_path).parent.as_posix()
        info_output_path = (
            Path(info_cfg.base_dir) / info_cfg.info_archive_dir / f"{info.name}.yaml"
        )
        info_output_path.parent.mkdir(parents=True, exist_ok=True)
        info_src_path = (
            Path(info_cfg.base_dir) / info_cfg.info_source_dir / f"{info.name}.yaml"
        )

        if os.path.exists(info_src_path):
            info_src = eKonf.load(info_src_path)
            available = info_src.get("available", True)

            for key in info_cfg.update_info:
                if key in info_src:
                    info[key] = info_src[key]
            for key, val in info_cfg.aggregate_info.items():
                if isinstance(info.splits, DictConfig):
                    vals = [
                        split[val] for split in info.splits.values() if val in split
                    ]
                else:
                    vals = [split[val] for split in info.splits if val in split]
                info[key] = sum(vals)
            for key, value in info_cfg.modified_info.items():
                if isinstance(info.splits, DictConfig):
                    vals = [
                        get_modified_time(f"{data_file_dir}/{split[value]}")
                        for split in info.splits.values()
                        if value in split
                    ]
                else:
                    vals = [
                        get_modified_time(f"{data_file_dir}/{split[value]}")
                        for split in info.splits
                        if value in split
                    ]
                vals = [v for v in vals if v is not None]
                if vals:
                    info[key] = max(vals)
            info["info_updated"] = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            # print(f'update info to {info_path}')
            eKonf.save(info, info_path)
            info_to_save = {}
            for key in info_cfg.info_list:
                if info.get(key) is not None:
                    info_to_save[key] = info[key]
            info_to_save = eKonf.to_config(info_to_save)

            if sample_output_dir:
                sample_text_path = Path(sample_output_dir) / f"{info.name}.txt"
            doc_output_path = (
                Path(info_cfg.base_dir) / info_cfg.info_output_dir / f"{info.name}.md"
            )
        else:
            available = False

        if available:
            infos.append(info_to_save)
            # print(f'saving info to {info_output_path}')
            eKonf.save(info_to_save, info_output_path)
            if sample_output_dir:
                save_sample_text(
                    data_file_dir, sample_text_path, info_cfg.sample_max_lines
                )

            src_link = f"{info_cfg.info_link_prefix}/{info.name}.yaml"
            sample_text_link = f"../sample/{info.name}.txt"
            fullname = info.get("fullname", info.name)
            # md_contents = markdown_template.format(name=info.fullname, src_path=src_path)
            if sample_output_dir:
                md_contents = markdown_template.format(
                    name=fullname, src_link=src_link, sample_text_link=sample_text_link
                )
            else:
                md_contents = markdown_template.format(name=fullname, src_link=src_link)
            doc_output_path.parent.mkdir(parents=True, exist_ok=True)
            # print(f'saving doc to {doc_output_path}')
            with open(doc_output_path, "w") as f:
                f.write(md_contents)
        else:
            if doc_output_path.exists():
                doc_output_path.unlink()
            if sample_output_dir:
                if sample_text_path.exists():
                    sample_text_path.unlink()

    builtins_path = Path(info_cfg.base_dir) / info_cfg.builtins_path
    builtins = eKonf.load(builtins_path)
    builtins[info_name] = [info.name for info in infos]
    eKonf.save(builtins, builtins_path)

    info_table = make_table(infos, table_cfg)
    readme_table_md = readme_md_template.format(info_table=info_table)
    readme_output_path = Path(info_cfg.base_dir) / info_cfg.readme_md_file
    readme_output_path.parent.mkdir(parents=True, exist_ok=True)
    # print(f'saving corpus to {corpus_output_path}')
    with open(readme_output_path, "w") as f:
        f.write(readme_table_md)

    info_fig_file = info_cfg.get("info_fig_file", None)
    if info_fig_file:
        fig_filepath = Path(info_cfg.base_dir) / info_fig_file
        make_figure(infos, fig_filepath)


def save_sample_text(data_file_dir, sample_text_path, sample_max_lines=None):
    sample_paths = glob(data_file_dir + "/sample-*.txt")
    if len(sample_paths) > 0:
        sample_path = sample_paths[0]
        with open(sample_text_path, "w") as f:
            with open(sample_path, "r") as sample_file:
                sample = sample_file.read().strip()
                sample = sample.split("\n")
                if sample_max_lines and len(sample) > sample_max_lines:
                    sample = sample[:sample_max_lines]
                f.write("\n".join(sample))


def make_figure(infos, fig_filepath):
    # import plotly.express as px
    import plotly.graph_objects as go

    values = []
    # KB = float(1024)
    # MB = float(KB ** 2)  # 1,048,576
    # GB = float(KB ** 3)  # 1,073,741,824

    total_size = sum([x.size_in_bytes for x in infos])
    total_size_en = sum([x.size_in_bytes for x in infos if x.lang == "en"])
    total_size_ko = sum([x.size_in_bytes for x in infos if x.lang == "ko"])
    for info in infos:
        name = info.name
        size = info.size_in_bytes
        quality = 1  # info.quality
        relative_weight = round(size * quality / total_size * 100, 2)
        values.append(
            [
                name,
                info.get("lang", "na"),
                "{:.2%}".format(relative_weight),
                relative_weight,
                size,
            ]
        )
    values.append(["Total", "", humanbytes(total_size, "GiB"), 0, total_size])
    values.append(["en", "Total", humanbytes(total_size, "GiB"), 0, total_size_en])
    values.append(["ko", "Total", humanbytes(total_size, "GiB"), 0, total_size_ko])

    headers = ["Name", "Language", "Relative_weight", "Weight", "Size"]
    df = pd.DataFrame(values, columns=headers)

    names = list(df.Name)
    weight = list(df.Weight)
    # sizes = [round(x / GB, 3) for x in list(df.Size)]
    langs = list(df.Language)
    # print(sizes)

    layout = go.Layout(
        paper_bgcolor="rgba(0,0,0,0)",
        # plot_bgcolor='rgba(0,0,0,0)'
    )
    fig = go.Figure(
        go.Treemap(
            labels=names,
            parents=langs,
            values=weight,
            textinfo="label+value",
            # values =  sizes,
            # textinfo = "label+value+percent parent+percent root",
            marker_colorscale="RdBu",
        ),
        layout=layout,
    )
    fig.update_layout(
        margin=dict(l=2, r=2, t=2, b=2),
    )

    fig.write_image(fig_filepath, scale=1.5)


def make_table(infos, table_cfg):

    df = pd.DataFrame(infos)
    eval_before = table_cfg.get("eval_before", None)
    if eval_before:
        for k, v in eval_before.items():
            df[k] = df.eval(v)
    sort_key = table_cfg.get("sort_key", None)
    if sort_key:
        df = df.sort_values(by=sort_key, ascending=False)
    calculate_weights_of = table_cfg.get("calculate_weights_of", None)
    if calculate_weights_of:
        for k, v in calculate_weights_of.items():
            df[k] = df[v].transform(lambda x: x / sum(x) * 100)
    make_links = table_cfg.get("make_links", None)
    if make_links:
        df[make_links.column] = df[make_links.column].apply(
            lambda x: make_links.link_format.format(x=x)
        )
    # df['weight'] = df['size_in_bytes'].transform(lambda x: x/sum(x)*100)
    dfs = [df]
    aggregate = table_cfg.get("aggregate", None)
    if aggregate:
        for agg in aggregate:
            group_by = agg["group_by"]
            sum_on = agg["sum"]
            names = agg["names"]
            if group_by is not None:
                df_grp = df.groupby(group_by)[sum_on].sum().reset_index()
                df_name = pd.DataFrame(names)
                df_grp = df_name.merge(df_grp, on=group_by)
                if calculate_weights_of:
                    for k, v in calculate_weights_of.items():
                        df_grp[k] = df_grp[v].transform(lambda x: x / sum(x) * 100)
            else:
                names.update({sum_on: int(df[sum_on].sum())})
                df_grp = pd.DataFrame([names])
            dfs.append(df_grp)

    df_table = pd.concat(dfs, axis=0)

    convert_to_humanbytes = table_cfg.get("convert_to_humanbytes", None)
    if convert_to_humanbytes:
        for col in convert_to_humanbytes:
            df_table[col] = df_table[col].transform(lambda x: humanbytes(x))
    format_columns = table_cfg.get("format_columns", None)
    if format_columns:
        for k, v in format_columns.items():
            df_table[k] = df_table[k].transform(lambda x: v.format(x=x))
    df_table.fillna("", inplace=True)

    headers = table_cfg.get("headers", None)
    if headers:
        df_table.rename(columns=headers, inplace=True)
        df_table = df_table[list(headers.values())]
    styles = table_cfg.get("styles", None)
    if styles:
        column_styles = [Style(**st) for st in styles]

    writer = MarkdownTableWriter(dataframe=df_table, column_styles=column_styles)
    writer.write_table()

    return writer.dumps()
