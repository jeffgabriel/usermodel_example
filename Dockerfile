FROM datarobot_dropin_sklearn

COPY requirements.txt .

RUN pip install -r requirements.txt