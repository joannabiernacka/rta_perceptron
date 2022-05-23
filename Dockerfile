FROM python:3.8-slim
WORKDIR /app_perc
ENV FLASK_APP=app_perceptron.py
ENV FLASK_RUN_HOST=0.0.0.0
COPY requirements.txt .
COPY app_perceptron.py .
COPY Perceptron.py .
COPY perceptron_model_iris.pkl .
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["flask", "run"]