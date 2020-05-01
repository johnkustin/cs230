#!/bin/bash

cat "Downloading Deepfake face dataset sample"
grep sample dataset-paths.csv | while read line
do
    kaggle datasets download $line
done

