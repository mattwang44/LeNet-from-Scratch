# This Python script is for auto downloading the files of MNIST dataset, provided by Yann LeCun (http://yann.lecun.com/exdb/mnist/).

# Reference:
# https://stackoverflow.com/a/16518224/7969188
# https://stackoverflow.com/questions/26577777/how-to-copy-and-extract-gz-files-using-python

from __future__ import ( division, absolute_import, print_function, unicode_literals )

import gzip
import os
import sys, os, tempfile, logging

if sys.version_info >= (3,):
    import urllib.request as urllib2
    import urllib.parse as urlparse
else:
    import urllib2
    import urlparse

def download(url, dest=None):
    
    """ 
    Download and save a file specified by url to dest directory,
    """
    u = urllib2.urlopen(url)

    scheme, netloc, path, query, fragment = urlparse.urlsplit(url)
    filename = os.path.basename(path)
    if not filename:
        filename = 'downloaded.file'
    if dest:
        filename = os.path.join(dest, filename)

    with open(filename, 'wb') as f:
        meta = u.info()
        meta_func = meta.getheaders if hasattr(meta, 'getheaders') else meta.get_all
        meta_length = meta_func("Content-Length")
        file_size = None
        if meta_length:
            file_size = int(meta_length[0])
        print("Downloading: {0} Bytes: {1}".format(url, file_size))

        file_size_dl = 0
        block_sz = 8192
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)

            status = "{0:16}".format(file_size_dl)
            if file_size:
                status += "   [{0:6.2f}%]".format(file_size_dl * 100 / file_size)
            status += chr(13)
            print(status, end="")
        print()

    return filename

def extract_gz(filename):
    with gzip.open(filename, 'rb') as infile:
        with open(filename[0:-3], 'wb') as outfile:
            for line in infile:
                outfile.write(line)

if __name__ == "__main__":
    url1 = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    url2 = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    url3 = "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    url4 = "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"

    if not os.path.isdir("MNIST"):
        os.mkdir("MNIST")
    os.chdir("./MNIST")

    print("Downloading MNIST dataset...")
    f1 = download(url1)
    f2 = download(url2)
    f3 = download(url3)
    f4 = download(url4)

    print("\nExtracting...")
    extract_gz("./"+f1)
    extract_gz("./"+f2)
    extract_gz("./"+f3)
    extract_gz("./"+f4)

    print("\nDeleting .gz files...")
    os.remove(f1)
    os.remove(f2)
    os.remove(f3)
    os.remove(f4)

    print("\nMNIST dataset is stored in the 'MNIST' folder")
