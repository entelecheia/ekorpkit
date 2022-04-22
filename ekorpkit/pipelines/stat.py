from ekorpkit.utils.func import humanbytes
from hydra.utils import instantiate
from ekorpkit.pipelines.pipe import apply
from omegaconf.listconfig import ListConfig


def summary_stats(
    df,
    key_columns=None,
    num_columns=None,
    agg_funcs=None,
    rename_columns=None,
    convert_to_humanbytes=None,
    num_workers=None,
    method=None,
    **kwargs,
):

    df = df.copy(deep=True)
    num_workers = num_workers if num_workers else 1
    text_keys = key_columns["text"]
    if isinstance(text_keys, (list, ListConfig)):
        for col in text_keys:
            df[col].fillna("", inplace=True)
            df[col] = df[col].astype(str)
        separator = "\n\n"
        text_key = text_keys[0]
        df[text_key] = df[text_keys].agg(separator.join, axis=1)
    else:
        text_key = text_keys
        df[text_key] = df[text_key].astype(str)

    for col, func in num_columns.items():
        print(f"apply {func} to {col}")
        len_func = instantiate(method[func])
        df[col] = apply(len_func, df[text_key])

    agg_funcs = {k: list(v) for k, v in agg_funcs.items()}
    df_sum = df.groupby(lambda _: True).agg(agg_funcs)
    df_sum.columns = ["_".join(x) for x in df_sum.columns.values]
    df_sum.rename(columns=dict(rename_columns), inplace=True)
    info = df_sum.to_dict(orient="records")[0]
    if convert_to_humanbytes:
        for k, v in convert_to_humanbytes.items():
            if k in info:
                info[v] = humanbytes(info[k])
    return info
