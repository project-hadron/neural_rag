import re, os, requests, configparser
from urllib.parse import unquote, urlparse


def get_filename_from_cd(cd):
    """
    Get filename from content-disposition
    """
    if not cd:
        return None
    filename = cd.split('filename=')[1]
    if filename.lower().startswith(("utf-8''", "utf-8'")):
        filename = filename.split("'")[-1]
    return unquote(filename)

def download_file(url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        filename = get_filename_from_cd(r.headers.get('content-disposition'))
        if not filename:
            filename = urlparse(url).geturl().replace('https://', '').replace('/', '-')
        filename = 'content/' + filename
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return filename

def get_extension(path):
    path = path.rstrip()
    path = path.replace(' \n', '')
    path = path.replace('%0A', '')
    if re.match(r'^https?://', path):
        filename = download_file(path)
    else:
        relative_path = path
        filename = os.path.abspath(relative_path)
    return os.path.splitext(filename)[1]

def getconfig(name: str=None, section: str=None) -> str:
    """Reads from a configuration file and section returning a dictionary of name value pairs
    by default the configuration name is 'config.ini' and the section 'main'. An example of a
    configuration file might be:

        [main]
        embedmodel=nomic-embed-text
        mainmodel=gemma:2b

    """
    name = name if isinstance(name, str) else 'config.ini'
    section = section if isinstance(section, str) else 'main'
    config = configparser.ConfigParser()
    config.read(name)
    return dict(config.items(section))