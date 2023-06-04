import pysrt

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
