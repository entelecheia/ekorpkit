import codecs
import simdjson
import orjson as json
import jsonpath_ng as jpath
import logging
from hydra.utils import instantiate
from omegaconf.listconfig import ListConfig
from ekorpkit.utils.func import any_to_utf8
from tqdm.auto import tqdm
from joblib import Parallel, delayed
from ekorpkit.utils.batch.batcher import tqdm_joblib


log = logging.getLogger(__name__)


parser = simdjson.Parser()


def _load_with_parser(parser, contents):
    if contents is None:
        return None
    contents = instantiate(parser, contents)
    return contents
    # try:
    # 	contents = instantiate(parser, contents)
    # 	return contents
    # except Exception:
    # 	msg.fail('Error parsing with parser {}'.format(parser))
    # 	# print(f'Error parsing {contents}')
    # 	return None


def json_loads(contents):
    try:
        return parser.parse(contents).as_dict()
    except ValueError:
        return contents


def load_json_to_list(contents, jpath_expression):
    contents = json.loads(contents)
    jpath_expression = jpath.parse(jpath_expression)
    results = []
    for match in jpath_expression.find(contents):
        results.append(match.value)
    return results


def load_json(contents):
    if contents is None:
        return None
    if len(contents) == 0:
        return None
    try:
        return [json.loads(contents)]
    except json.JSONDecodeError:
        log.critical(f"Error parsing {contents}")
        raise Exception


def load_jsonlines(contents):
    lines = contents.split("\n")
    # print(f'Loading jsonlines from {len(lines)} lines')
    data = []
    for lineno, line in enumerate(lines):
        if len(line) == 0:
            continue
        js = None
        try:
            js = json.loads(line)
        except json.JSONDecodeError as err:
            log.warning(f"Skipping lineno {lineno} - colno:{err.colno} msg:{err.msg}")
        if js:
            data.append(js)

    return data


def parse_data(contents, parse_args, default_items, num_workers=1):
    parser = parse_args.get("parser", None)
    decompressor = parse_args.get("decompressor", None)
    filename = default_items.get("filename", "")

    if decompressor is not None and ".gz" in filename:
        contents = instantiate(decompressor, contents)

    decode_before_parse = parse_args.get("decode_before_parse", False)
    multiprocessing_at = parse_args.get("multiprocessing_at", "load_data")

    documents = []
    if decode_before_parse and isinstance(contents, (bytes, bytearray)):
        try:
            contents = any_to_utf8(contents)
            # contents = ftfy.fix_text(any_to_utf8(contents))
        except Exception:
            log.warning(f"Decoding error")
            print(contents)
            return documents

    if parser is not None and parser.get("_target_", None):
        contents = _load_with_parser(parser, contents)
    else:
        contents = load_json(contents)
    if contents is None:
        return documents
    if len(contents) == 0:
        return documents

    if isinstance(contents, str):
        contents = [contents]

    if num_workers > 1 and multiprocessing_at == "parse_data":
        log.info(
            f"Starting multiprocessing with {num_workers} processes at {multiprocessing_at}"
        )
        desciption = default_items.get("filename", "::parse_data()")
        with tqdm_joblib(tqdm(desc=desciption, total=len(contents))) as pbar:
            results = Parallel(n_jobs=num_workers)(
                delayed(parse_json)(content, parse_args, default_items, 1)
                for content in contents
                if len(content) > 0
            )
            for result in results:
                documents += result
    else:
        if len(contents) > 1000:
            log.info("Number of data in the contents: {}".format(len(contents)))
        for i, content in enumerate(contents):
            # print(content)
            if len(content) > 0:
                documents += parse_json(content, parse_args, default_items, num_workers)

    return documents


def parse_json(data, parse_args, default_items={}, num_workers=1):

    documents = []
    meta_args = parse_args.get("meta", {})
    meta_field = meta_args.get("field", None)
    data_args = parse_args["data"]
    data_field = data_args.get("field", None)

    if meta_args.get("item", None) is not None:
        meta_data = data[meta_field] if meta_field else data
        meta = data_to_document(meta_data, meta_args)
        if len(meta) > 0:
            default_items.update(meta[0])

    data = data[data_field] if data_field else data
    if isinstance(data, list):
        totalcount = len(data)
        if num_workers > 1 and totalcount > num_workers:
            log.info(
                f"Starting multiprocessing with {num_workers} processes at parse_json"
            )
            desciption = default_items.get("filename", "::parse_json()")
            with tqdm_joblib(tqdm(desc=desciption, total=totalcount)) as pbar:
                results = Parallel(n_jobs=num_workers)(
                    delayed(data_to_document)(doc, data_args, default_items)
                    for doc in data
                )
                for result in results:
                    documents += result
        else:
            if totalcount > 1000:
                log.info("Total number of documents: {}".format(totalcount))
            for ix, doc in enumerate(data):
                documents += data_to_document(doc, data_args, default_items)
    else:
        documents += data_to_document(data, data_args, default_items)

    return documents


def data_to_document(data, data_args, default_items={}):
    documents = []
    # data.update(default_items)

    item_separator = data_args.get("item_separator", "")
    item_separator = codecs.decode(item_separator, "unicode_escape")

    document = {}
    for k, v in data_args["item"].items():
        val = get_item_value(data, v)
        if isinstance(val, list):
            strval = []
            for sv in val:
                if sv:
                    if len(str(sv).strip()) > 0:
                        strval.append(str(sv).strip())
                # else:
                # 	# strval.append('')
                # 	print(v)
            val = item_separator.join(strval)
        document[k] = val
    document.update(default_items)
    documents.append(document)
    return documents


def get_item_value(item, key):
    if item is None or key is None:
        return None
    try:
        if isinstance(key, str):
            if "." in key:
                return get_value_by_jpath(item, key)
            else:
                return item[key] if key in item else None
        elif isinstance(key, (list, ListConfig)):
            value = []
            for k in key:
                val = get_item_value(item, k)
                if isinstance(val, list):
                    value += val
                else:
                    value.append(val)
            return value
        else:
            print(key, type(key))
            raise ValueError("key must be str or list")
    except Exception as e:
        log.critical(item)
        log.critical(key, type(key))
        print(e)
        raise e
        # return None


def get_value_by_jpath(item, expression):
    jpath_expression = jpath.parse(expression)
    results = []
    for match in jpath_expression.find(item):
        if isinstance(match.value, list):
            val = "\n".join(match.value)
        else:
            val = match.value
        results.append(val)
    if len(results) == 0:
        return None
    elif len(results) == 1:
        return results[0]
    else:
        return results
