#!/bin/bash

APP_PID=
stopRunningProcess() {
    if test ! "${APP_PID}" = '' && ps -p ${APP_PID} > /dev/null ; then
        > /proc/1/fd/1 echo "Stopping ${COMMAND_PATH} which is running with process ID ${APP_PID}"
        kill -TERM ${APP_PID}
        > /proc/1/fd/1 echo "Waiting for ${COMMAND_PATH} to process SIGTERM signal"
        wait ${APP_PID}
        > /proc/1/fd/1 echo "All processes have stopped running"
    else
        > /proc/1/fd/1 echo "${COMMAND_PATH} was not started when the signal was sent or it has already been stopped"
    fi
}

trap stopRunningProcess EXIT TERM

#HF workaround flags (ditch Xsrf flag locally)

streamlit run ${HOME}/blueprint/demo/app.py \
    --server.address=0.0.0.0 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false &

APP_PID=${!}

wait ${APP_PID}