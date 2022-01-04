#!/usr/bin/env bash

if [[ "$1" = "train" ]]; then
     nohup /usr/sbin/sshd -D &
     CURRENT_HOST=$(jq .current_host /opt/ml/input/config/resourceconfig.json)
     sed -ie "s/PLACEHOLDER_HOSTNAME/$CURRENT_HOST/g" /opt/ml/code/changehostname.c
     gcc -o /opt/ml/code/changehostname.o -c -fPIC -Wall /opt/ml/code/changehostname.c
     gcc -o /opt/ml/code/libchangehostname.so -shared -export-dynamic /opt/ml/code/changehostname.o -ldl
     LD_PRELOAD=/opt/ml/code/libchangehostname.so /opt/ml/code/train
else
     eval "$@"
fi
