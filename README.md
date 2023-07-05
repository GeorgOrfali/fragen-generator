# Automatische Aufgaben generierung mittels NLP
Für mein Bachelorprojekt habe ich ein Machine Learning Algorithmus entwickelt welcher aus Vorlesungsfolien, automatisch Fragen für die Lehre generiert.

# Set Up
Die Software nutzt Python version: 3.7.15

## Dependencies herunterladen
`
make setup
`

### Falls pipenv nicht vorhanden
`
pip install pipenv
`
## Software starten
### zuerst Pipenv shell starten
`
make shell
`

### Flask server starten
`
make run
`

### Projekt Testen
Dazu gehe auf den Link um Wahr-Falsch und LückenText Aufgaben zu testen:
`
http://127.0.0.1:5000/
`
Um Zusätzlich Single Choice mittels RNN zu generieren und alle anderen Wahr-Falsch und LückenText Aufgaben zu generieren:
`
http://127.0.0.1:5000/all
`