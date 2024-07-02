#!/bin/bash

set -e

if ! command -v python3 &> /dev/null; then
  echo "Python3 is not installed. Skipping HeavyIQ deployment."
  exit 0
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
PYTHON_MAJOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR_VERSION=$(echo $PYTHON_VERSION | cut -d. -f2)

if [[ $PYTHON_MAJOR_VERSION -eq 3 && $PYTHON_MINOR_VERSION -ge 10 ]]; then
  if [[ -f "$CONFIG_FILE" ]]; then
    CONFIG_FILE_ABS_PATH=$(readlink -f $CONFIG_FILE)
  fi
  HEAVYIQ_LOG_PREFIX=$(readlink -f $MAPD_DATA)/log/heavyiq
  pushd heavyiq > /dev/null
  if [[ ! -d .venv ]]; then
    BUILD_LOG_FILE=${HEAVYIQ_LOG_PREFIX}_build.log
    VENV_ERROR=false
    python3 -m venv .venv &>> $BUILD_LOG_FILE || VENV_ERROR=true
    if $VENV_ERROR; then
      echo "Warning: An error occurred when attempting to create a virtual environment." \
           "See the $BUILD_LOG_FILE logs for more details."
      rm -rf .venv
      # Continue running HeavyDB and the web server, even when an error occurs for HeavyIQ.
      exit 0
    fi
    source .venv/bin/activate

    echo "Installing HeavyIQ dependencies."
    PIP_ERROR=false
    REQUIREMENTS=requirements.txt
    if [[ -f requirements.packages.txt ]] ; then
      REQUIREMENTS=requirements.packages.txt
    fi
    pip install -r $REQUIREMENTS &>> $BUILD_LOG_FILE || PIP_ERROR=true
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
  if [ "${START_MODE}" == "f" ]; then
    gunicorn --preload --log-file $GUNICORN_LOG_FILE --capture-output -t 0 -w 4 -k uvicorn.workers.UvicornWorker \
    -b :$MAPD_HEAVYIQ_PORT -c "gunicorn.conf.py" "heavyiq.api:create_app(config_path=\"$CONFIG_FILE_ABS_PATH\")"
  else
    gunicorn --preload --log-file $GUNICORN_LOG_FILE --capture-output -t 0 -w 4 -k uvicorn.workers.UvicornWorker \
    -b :$MAPD_HEAVYIQ_PORT -c "gunicorn.conf.py" "heavyiq.api:create_app(config_path=\"$CONFIG_FILE_ABS_PATH\")" &
  fi
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
  echo "Python 3.10 not found or installed version is less than 3.10. Skipping HeavyIQ deployment."
fi
