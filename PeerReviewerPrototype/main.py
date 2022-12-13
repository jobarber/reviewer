import argparse

from peerreviewer.reviewer import Document


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_docx", help="the path to your docx file", required=True)
    parser.add_argument("--path_to_output", help="where to place the output file", default='output.docx')
    args = parser.parse_args()

    document = Document(path_to_docx=args.path_to_docx)
    document.create_analysis()
    document.save(args.path_to_output)
