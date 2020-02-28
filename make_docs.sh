#!/usr/bin/env bash

if [ ! -d "docs" ]; then
  (
    mkdir "docs"
    cd docs
    sphinx-quickstart
    sphinx-build -b html vailtools docs/_build
  )
fi

sphinx-apidoc -o docs/source vailtools
cd docs
make html
