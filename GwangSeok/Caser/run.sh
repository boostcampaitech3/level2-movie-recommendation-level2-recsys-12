#!/bin/sh

# nohup python -u {파일 이름}.py > {파일 이름}.log & -> 지정된 파이썬 파일을 실행하고 실행 출력은 log 파일로 저장
# ps -ef | grep {파일 이름}.py -> 프로세스가 정상적으로 돌아가고 있나 확인할 때
# tail -f {파일 이름}.log -> 출력 결과 실시간으로 확인할때

# pwd=$(readlink -f .)

nohup /opt/conda/bin/python -u /opt/ml/competition/GwangSeok/Caser/train.py --d=100 --save_metric=recall --batch_size=256 --l2=1e-3 --num_valid_item=10 > /opt/ml/competition/GwangSeok/Caser/output/trainig.log && \
nohup /opt/conda/bin/python -u /opt/ml/competition/GwangSeok/Caser/inference.py --d=100 --save_metric=recall > /opt/ml/competition/GwangSeok/Caser/output/inference.log &