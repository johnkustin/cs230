#!/bin/bash

cat "Downloading Deepfake face dataset"
cat dataset-paths.csv | while read line
do
    echo "Downloading $line"
    kaggle datasets download $line
done

