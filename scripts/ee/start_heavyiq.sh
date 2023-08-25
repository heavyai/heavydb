#!/bin/bash

set -e

HAS_PYTHON_3_10=$(hash python3.10 2> /dev/null || echo "false")
if [[ "$HAS_PYTHON_3_10" != "false" ]]; then
  if [[ -f "$CONFIG_FILE" ]]; then
    CONFIG_FILE_ABS_PATH=$(readlink -f $CONFIG_FILE)
  fi
  HEAVYIQ_LOG_PREFIX=$(readlink -f $MAPD_DATA)/log/heavyiq
  pushd heavyiq > /dev/null
  if [[ ! -d .venv ]]; then
    BUILD_LOG_FILE=${HEAVYIQ_LOG_PREFIX}_build.log
    VENV_ERROR=false
    python3.10 -m venv .venv &>> $BUILD_LOG_FILE || VENV_ERROR=true
    if $VENV_ERROR; then
      echo "Warning: An error occurred when attempting to create a virtual environment." \
           "See the $BUILD_LOG_FILE logs for more details."
      rm -rf .venv
      # Continue running HeavyDB and the web server, even when an error occurs for HeavyIQ.
      exit 0
    fi
    source .venv/bin/activate

    PIP_ERROR=false
    pip install -r requirements.txt &>> $BUILD_LOG_FILE || PIP_ERROR=true
    if $PIP_ERROR; then
      echo "Warning: An error occurred when installing HeavyIQ dependencies." \
           "See the $BUILD_LOG_FILE logs for more details."
      rm -rf .venv
      # Continue running HeavyDB and the web server, even when an error occurs for HeavyIQ.
      exit 0
    fi
  else
    source .venv/bin/activate
  fi
  GUNICORN_LOG_FILE=${HEAVYIQ_LOG_PREFIX}_gunicorn.log
  gunicorn --preload --log-file $GUNICORN_LOG_FILE --capture-output -t 0 -w 4 -k uvicorn.workers.UvicornWorker \
  -b :$MAPD_HEAVYIQ_PORT "heavyiq.api:create_app(config_path=\"$CONFIG_FILE_ABS_PATH\")" &
  PID3=$!
  sleep 5
  if [[ $(ps -h -p $PID3) ]]; then
    echo "HeavyIQ HTTP:  localhost:${MAPD_HEAVYIQ_PORT}"
    echo "- heavy_iq $PID3 started"
  else
    echo "Warning: An error occurred when starting up HeavyIQ. See the $GUNICORN_LOG_FILE logs for more details."
    # Continue running HeavyDB and the web server, even when an error occurs for HeavyIQ.
    exit 0
  fi
  popd > /dev/null
else
  echo "Python 3.10 not found. Skipping HeavyIQ deployment."
fi
