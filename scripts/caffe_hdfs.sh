#!/bin/bash

# Script to launch caffe executable with support to hdfs
# Use Caffe_hdfs.sh instead of caffe.bin or caffe executable

ROOTDIR=$(dirname $0)
source $ROOTDIR/devenv.sh

$ROOTDIR/caffe.bin ${@:1}



