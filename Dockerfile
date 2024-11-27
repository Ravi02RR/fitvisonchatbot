FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
EXPOSE 5000
ENV FLASK_APP=src/smallmode.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]