#!/bin/zsh
# 건축물대장 이어받기(하루 호출 한도 때문에 매일 0시 5분 cron으로 실행). 다 받으면 parcel_test.py까지 돌리고 결과를 logs/에 남김
cd "$(dirname "$0")/.." || exit 1
D=$(date +%Y%m%d_%H%M)
.venv/bin/python code/parcel_fetch.py > "logs/parcel_fetch_$D.log" 2>&1
.venv/bin/python code/parcel_test.py > "logs/parcel_test_$D.log" 2>&1
