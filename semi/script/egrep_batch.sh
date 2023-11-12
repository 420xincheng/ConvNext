#!/bin/bash


file_pattern=$1
regex=$2

for f in $file_pattern; do
    echo $f
    egrep $regex $f | tail -n 1
    echo 
done


