FROM python:3.11-slim

WORKDIR /app

COPY detector/requirements.txt /app/detector/requirements.txt
RUN pip install --no-cache-dir -r /app/detector/requirements.txt

COPY detector /app/detector
COPY data /app/data

EXPOSE 8000

CMD ["uvicorn", "detector.detector:app", "--host", "0.0.0.0", "--port", "8000"]
