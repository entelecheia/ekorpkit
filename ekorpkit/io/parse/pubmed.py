import pubmed_parser


def parse_medline_xml(contents):
    try:
        return pubmed_parser.parse_medline_xml(contents)
    except Exception as e:
        print(e)
        return None


def parse_pubmed_paragraph(contents):
    try:
        return pubmed_parser.parse_pubmed_paragraph(contents, all_paragraph=False)
    except Exception as e:
        print(e)
        return None
