if [ -z "$1" ]
then
    echo "Inserisci nome del folder di interesse"
    exit 1
fi

docker run -v /raid/ireneamerini/lorenzo_kdxai/SimKD/:/work/project/  --gpus all --ipc host -u 1001:1001 lorenzopapa5/cuda11.3-python3.8-pytorch1.11 /usr/bin/python3 /work/project/json_to_tex.py -path "${1}"