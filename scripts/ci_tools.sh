#!/usr/bin/env bash


function human() {
    printf "\n\nIdentifying potential security vulnerabilities with bandit.\n"
    python -m bandit -r vailtools

    printf "\n\nLocating dead code with vulture.\n"
    python -m vulture vailtools
}


function non_human() {
    printf "\n\nOrganizing imports with isort.\n"
    python -m isort vailtools/**/*.py tests/**/*.py

    printf "\n\nFormatting code with black.\n"
    python -m black vailtools tests
}


# This will ensure relative paths will work as expected
# cd to the directory where this script resides
cd "$(dirname "$(readlink -f "$0")")" || exit
# Then to the top level of the package
cd ..

# Handle optional arguments
MODE="${1:-both}"

# Dispatch to the correct functions
if [ "$MODE" = "human" ]; then
    human

elif [ "$MODE" = "non-human" ]; then
    non_human

elif [ "$MODE" = "both" ]; then
    non_human
    human

else
    echo "$MODE is not a vaild mode, expected one of {human, non-human, both}."
    exit
fi
