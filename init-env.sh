#!/bin/bash

# Remove existing venv folder if it exists
if [ -d "venv" ]; then
	rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r ./requirements.txt
