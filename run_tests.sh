#!/usr/bin/env bash

# use the -s flag to see std out, useful for viewing model summaries
python -m pytest --cov=vailtools/ tests/ -v