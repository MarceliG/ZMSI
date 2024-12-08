# ZMSI - Zastosowanie Metod Sztucznej Inteligencji

## Setting up the environment

1. `python3 -m venv .venv`
2. `source .venv/bin/activate`
3. `pip install poetry`
4. `poetry install`
    - If you want create new `lock` file use command `poetry lock`

## Comand
Before pushing any code to a remote repository, check locally if everything is formatted and typed:
1. `poetry run ruff check --diff`
2. `poetry run ruff format --diff`
3. `poetry run mypy .`

Run application:

`poetry run python3 main.py`

Run web application

`poetry run python3 app/manage.py runserver`
