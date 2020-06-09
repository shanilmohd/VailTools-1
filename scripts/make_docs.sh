#!/usr/bin/env bash

# This will ensure relative paths will work as expected
# cd to the directory where this script resides
cd "$(dirname "$(readlink -f "$0")")" || exit
# Then to the top level of the package
cd ..

if [ ! -d docs ]; then
  (
    mkdir docs
    cd docs || exit
    sphinx-quickstart
  )
fi

sphinx-apidoc -M -f -o docs/source vailtools
sphinx-build -b html docs/source docs/_build
