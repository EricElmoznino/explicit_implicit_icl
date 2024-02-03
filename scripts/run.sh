#!/usr/bin/env bash

command=$1

for seed in {0..4}; do
    $command seed=$seed &
done
wait