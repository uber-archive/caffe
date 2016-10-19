#!/bin/bash

# Script to launch caffe executable with support to hdfs
# Use Caffe_hdfs.sh instead of caffe.bin or caffe executable

# Setup Java
export JAVA_HOME=$(readlink -f /usr/bin/java | sed "s:bin/java::")
export LD_LIBRARY_PATH=$JAVA_HOME/lib/amd64/server/:$LD_LIBRARY_PATH

# Setup HDFS
export HADOOP_PREFIX=$(dirname $(dirname $(which hdfs)))
for jar in `find $HADOOP_PREFIX/share/hadoop/hdfs/ -name \*jar`; do export CLASSPATH=$CLASSPATH:$jar; done
for jar in `find $HADOOP_PREFIX/share/hadoop/common/ -name \*jar`; do export CLASSPATH=$CLASSPATH:$jar; done
export LD_LIBRARY_PATH=$HADOOP_PREFIX/lib/native:$LD_LIBRARY_PATH

$(dirname $0)/caffe.bin ${@:1}



