# Steps to package codes to docker and run from root directory

FROM python:3.12

RUN pip install -U pip
RUN pip install pipenv 

WORKDIR /app

COPY ["Pipfile", "Pipfile.lock", "./"]

RUN pipenv install --system --deploy

COPY ["deployment/scoring.py", "./"]

ENTRYPOINT ["python", "./scoring.py"]