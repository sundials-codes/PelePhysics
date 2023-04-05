#!/usr/bin/env bash

MECH_HOME="$(pwd)"
MECH_FILE="${MECH_HOME}/qssa.yaml"
bash ../converter.sh -f "${MECH_FILE}"
