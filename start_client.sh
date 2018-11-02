#!/usr/bin/env bash

SERVER_IP=127.0.0.1
SERVER_PORT=1234

python -u web-demo/app.py \
    --server_ip ${SERVER_IP} \
    --server_port ${SERVER_PORT}