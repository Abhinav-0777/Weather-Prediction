FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y git

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["uvicorn","src.api.app:app"]

CMD ["--host","0.0.0.0","--port","9000"]

