FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["uvicorn","app:app"]

CMD ["--host","0.0.0.0","--port","9000"]

