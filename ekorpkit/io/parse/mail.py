from wasabi import msg
from .json import load_json


def mail_to_json(contents):
    try:
        import mailparser
    except ImportError:
        msg.warn(
            "mailparser is not installed. Please install it with `pip install mailparser`"
        )
    try:
        contents = mailparser.parse_from_string(contents).mail_json
    except Exception as err:
        print(f"{contents}")
        print(err)
        contents = None
    return load_json(contents)
