#!/usr/bin/env bash

MECH_HOME="$(pwd)"
MECH_FILE="${MECH_HOME}/skeletal.yaml"
bash ../converter.sh -f "${MECH_FILE}"
