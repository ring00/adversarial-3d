#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import urllib
import tarfile
from config import cfg

def download_and_extract(URL, DIR):
    filename = URL.split('/')[-1]
    filepath = os.path.join(DIR, filename)
    if not os.path.exists(DIR):
        os.mkdir(DIR)
    extract_to = os.path.splitext(filepath)[0]

    def _progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading {} {:.1f}%".format(filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    if os.path.exists(filepath):
        print("file {} already exist".format(filename))
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, _progress)
        print()
        print('Successfully downloaded', filename, os.path.getsize(filepath), 'bytes.')
        tarfile.open(filepath, 'r:gz').extractall(cfg.model_dir)

if __name__ == '__main__':
    download_and_extract(cfg.model_url, cfg.model_dir)
