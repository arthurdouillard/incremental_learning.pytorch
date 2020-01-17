import gzip
import io
import logging
import os
import urllib.request

logger = logging.getLogger(__name__)

URLS = {
    "googlenews":
    "https://github.com/eyaler/word2vec-slim/raw/master/GoogleNews-vectors-negative300-SLIM.bin.gz"
}


def fetch_word_embeddings(folder, name="googlenews"):
    if name == "googlenews":
        return _fetch_googlenews_word2vec(folder)
    raise ValueError("Unknown embedding type {}.".format(name))


def _fetch_googlenews_word2vec(folder):
    output_file = os.path.join(folder, "googlenews.bin")

    if os.path.exists(output_file):
        logger.info("googlenews.bin already exist! Skipping.")
        return output_file

    response = urllib.request.urlopen(URLS["googlenews"])

    logger.info("Downloading googlenews...")
    compressed_file = io.BytesIO(response.read())

    logger.info("Decompressing googlenews...")
    decompressed_file = gzip.GzipFile(fileobj=compressed_file, mode='rb')

    with open(output_file, 'wb+') as f:
        f.write(decompressed_file.read())

    return output_file
