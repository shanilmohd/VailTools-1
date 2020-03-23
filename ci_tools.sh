#!/usr/bin/env bash


function human() {
    printf "\n\nIdentifying potential security vulnerabilities with bandit.\n"
    python -m bandit -r vailtools

    printf "\n\nLocating dead code with vulture.\n"
    python -m vulture vailtools
}


function non_human() {
    printf "\n\nOrganizing imports with isort.\n"
    python -m isort setup.py vailtools/**/*.py tests/**/*.py

    printf "\n\nFormatting code with black.\n"
    python -m black setup.py vailtools tests
}


MODE="${1:-both}"

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
