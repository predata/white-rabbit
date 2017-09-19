#!/bin/bash
set -e

if [[ $# -ne 1 ]]; then
  echo "Usage: ./upload.sh [version]"
  exit 1
fi

VERSION=$1

echo "Building source distribution"
python setup.py sdist

echo "Building binary distribution"
python setup.py bdist

twine upload "dist/white_rabbit-${VERSION}.tar.gz"
