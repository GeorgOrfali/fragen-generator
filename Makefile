setup:
	pipenv install

shell:
	pipenv shell

run:
	flask --app src/routes.py --debug run

train:
	python src/train_RNN.py