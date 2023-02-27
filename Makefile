all:
    pipenv install
    pipenv shell
    flask --app src/routes.py --debug run