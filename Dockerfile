FROM kennethreitz/pipenv:latest
COPY . /app
CMD python3 ./app.py
