#!/bin/bash


job_name=$1
command=$2

mkdir -p logs/log-new

$command 2>&1|tee -a logs/log-new/$job_name.log
