#!/bin/sh
# Downloads MNIST into data/ at the top of the workspace, checks the files against the
# checksums torchvision uses, and uncompresses them (Chapter 19). About 55 MB unpacked.
# ember itself uses only the standard library, which has no gzip decoder, so this is a script.
set -eu

cd "$(dirname "$0")/.."
mkdir -p data
cd data

mirror=https://ossci-datasets.s3.amazonaws.com/mnist

fetch() { # fetch <file> <md5 of file.gz>
    if [ -f "$1" ]; then
        echo "$1: already here"
        return
    fi
    curl --fail --location --silent --show-error -o "$1.gz" "$mirror/$1.gz"
    if command -v md5sum >/dev/null 2>&1; then
        sum=$(md5sum "$1.gz" | cut -d ' ' -f 1)
    else
        sum=$(md5 -q "$1.gz") # macOS
    fi
    if [ "$sum" != "$2" ]; then
        echo "$1.gz: checksum is $sum, expected $2" >&2
        rm -f "$1.gz"
        exit 1
    fi
    gunzip "$1.gz"
    echo "$1: ok"
}

fetch train-images-idx3-ubyte f68b3c2dcbeaaa9fbdd348bbdeb94873
fetch train-labels-idx1-ubyte d53e105ee54ea40749a09fcbcd1e9432
fetch t10k-images-idx3-ubyte 9fb629c4189551a2d022fa330f9573f3
fetch t10k-labels-idx1-ubyte ec29112dd5afa0611ce80d1b7f02629c
