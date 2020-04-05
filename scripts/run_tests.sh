#!/usr/bin/env bash

# This will ensure relative paths will work as expected
# cd to the directory where this script resides
cd "$(dirname "$(readlink -f "$0")")" || exit
# Then to the top level of the package
cd ..

# use the -s flag to see std out, useful for viewing model summaries
python -m pytest --lf --cov=vailtools/ tests/ -v
