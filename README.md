# Automatische Aufgaben generierung mittels NLP
Für mein Bachelorprojekt habe ich ein Machine Learning Algorithmus entwickelt welcher aus Vorlesungsfolien, automatisch Fragen für die Lehre generiert.

# Set Up
Die Software nutzt Python version: 3.7.15

## Dependencies herunterladen
`
pipenv install
`

### Falls pipenv nicht vorhanden
`
pip install pipenv
`
## Software starten
### zuerst Pipenv shell starten
`
pipenv shell
`

### Flask server starten
`
flask --app src/routes.py --debug run
`