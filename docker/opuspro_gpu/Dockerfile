#caffe-hdfsgpu-2-0-0
#latest
#../..

# build this docker:
# opus-build-test -R localhost:5000 -i caffe-hdfsgpu-2-0-0 --push -l DEBUG -d test .


FROM localhost:5000/gpu-base:caffebuild-test-2016_10_20_11_24_45

RUN apt-get update && \
    apt-get install -y git libhdf5-serial-dev protobuf-compiler && \
    cd /usr/lib/x86_64-linux-gnu/ && \
    ln -s libhdf5_serial.so.8.0.2 libhdf5.so && \
    ln -s libhdf5_serial_hl.so.8.0.2 libhdf5_hl.so


RUN mkdir /opt/caffe-hdfs/
ADD . /opt/caffe-hdfs/.

#RUN git clone https://github.com/uber-archive/caffe && \
#    mv caffe /opt/caffe-hdfs

#####################
# get hadoop native libraries
#####################
RUN mkdir $HADOOP_PREFIX/lib && \
    mkdir $HADOOP_PREFIX/lib/native && \
    cp /opt/caffe-hdfs/docker/opusprod/native/* $HADOOP_PREFIX/lib/native/. && \
    cd $HADOOP_PREFIX/lib/native/. && \
    for jar in `find $HADOOP_PREFIX/share/hadoop/hdfs/ -name \*jar`; do export CLASSPATH=$CLASSPATH:$jar; done && \
    for jar in `find $HADOOP_PREFIX/share/hadoop/common/ -name \*jar`; do export CLASSPATH=$CLASSPATH:$jar; done

ENV LD_LIBRARY_PATH $HADOOP_PREFIX/lib/native:$JAVA_HOME/jre/lib/amd64/server/:$LD_LIBRARY_PATH

RUN cd /opt/caffe-hdfs && \
    cp Makefile.config.docker Makefile.config && \
    make distribute
ENV LD_LIBRARY_PATH /opt/caffe-hdfs/distribute/lib:$LD_LIBRARY_PATH
ENV PATH /opt/caffe-hdfs/distribute/bin:$PATH
ENV PYTHONPATH /opt/caffe-hdfs/distribute/python:$PYTHONPATH