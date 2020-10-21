FROM gcr.io/deeplearning-platform-release/pytorch-gpu
WORKDIR /app
ADD . /app
RUN pip3 install -r requirements.txt
EXPOSE 8080
CMD ["python3", "/app/main.py"]
