#! /bin/bash

export FLASK_APP=src/server
export FLASK_ENV=development
export FLASK_RUN_PORT=5000
flask run --host=0.0.0.0
