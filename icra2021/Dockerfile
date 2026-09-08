FROM python:3.7.17-slim-bullseye
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2
WORKDIR /app
COPY icra2021/requirements.txt /tmp/icra-requirements.txt
RUN python -m pip install --no-cache-dir pip==23.3.2 setuptools==65.6.3 wheel==0.38.4 && python -m pip install --no-cache-dir -r /tmp/icra-requirements.txt
COPY icra2021 /app/icra2021
RUN python -m icra2021.bootstrap --checkpoint
CMD ["python", "-m", "icra2021.check", "--tensorflow"]
