#!/bin/bash

MODEL_SEED=0
SRUNNER_SEED=0

export LEADERBOARD_ROOT="leaderboard"
export SRUNNER_ROOT="scenario_runner_seed${SRUNNER_SEED}"
export CARLA_ROOT=carla
export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=~/eval_my:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${LEADERBOARD_ROOT}
export PYTHONPATH=$PYTHONPATH:${SRUNNER_ROOT}
export PYTHONPATH=$PYTHONPATH:team_code

MODEL_NAME="${MODEL_SEED}_${SRUNNER_SEED}"

export CHECKPOINT_ENDPOINT="result_${MODEL_NAME}.json"
export SCENARIOS=${LEADERBOARD_ROOT}/data/all_towns_traffic_scenarios_public.json
export ROUTES=routes_town05_long.xml
export PORT=20000
export TM_PORT=$((PORT + 500))
export HOST=localhost

export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export REPETITIONS=1
export TEAM_AGENT=team_code/agent.py
export TEAM_CONFIG=ckpt/main_1.ckpt
export RESUME=True

export SAVE_PATH=data/${MODEL_NAME}

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
--scenarios=${SCENARIOS}  \
--routes=${ROUTES} \
--repetitions=${REPETITIONS} \
--track=${CHALLENGE_TRACK_CODENAME} \
--checkpoint=${CHECKPOINT_ENDPOINT} \
--agent=${TEAM_AGENT} \
--agent-config=${TEAM_CONFIG} \
--debug=${DEBUG_CHALLENGE} \
--record=${RECORD_PATH} \
--resume=${RESUME} \
--port=${PORT} \
--trafficManagerPort=${TM_PORT}

