import pysrt
import pathlib

import tempfile


def save_srt(file_name, data):
    temp_file = tempfile.NamedTemporaryFile(suffix=file_name, delete=False)
    temp_file_path = temp_file.name
    temp_file.write(data)
    return temp_file_path


def subs_text(path):
    try:
        srt = pysrt.open(path)
    except UnicodeDecodeError:
        srt = pysrt.open(path, encoding='iso-8859-1')
    text = srt.text
    return text


def process_path(path):
    filename = pathlib.Path(path).stem
    level = pathlib.Path(path).parent.name
    if level not in ('A1', 'A2', 'B1', 'B2', 'C1'):
        level = None
    text = subs_text(path)
    return filename, text, level
