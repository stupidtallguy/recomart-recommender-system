FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-api-lock.txt requirements-demo-lock.txt ./
RUN python -m pip install --upgrade pip \
    && python -m pip install -r requirements-api-lock.txt -r requirements-demo-lock.txt

COPY . .

RUN useradd --create-home --shell /usr/sbin/nologin recomart \
    && chown -R recomart:recomart /app

USER recomart

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
