import logging
from .json import load_json


log = logging.getLogger(__name__)


def mail_to_json(contents):
    try:
        import mailparser
    except ImportError:
        log.warning(
            "mailparser is not installed. Please install it with `pip install mailparser`"
        )
    try:
        contents = mailparser.parse_from_string(contents).mail_json
    except Exception as err:
        print(f"{contents}")
        print(err)
        contents = None
    return load_json(contents)
