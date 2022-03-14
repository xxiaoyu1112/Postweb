实验室GO环境相关资料

go 安装位置 /usr/local/go

已经安装了grpc

consul 是一个xDS软件，可以作服务发现

envoy 是一个代理软件，包括负载均衡等策略



#### 数据库设计

|id| task_id| uid|time_tag|region_tag|arrive_time|region_type|

```
下单时间 
经度 
纬度 
区块类型 
与所在订单的直线相对距离  
与前一单的直线绝对距离 
预约-接单 
预约-当前
```



day_tag,

20210401,13584,上海市,2208775596316,2021-04-01 08:57:20,2021-04-01 11:00:00,2021-04-01 13:00:00,121.513188,31.308708000000006,24567482,3110,居民区,2021-04-01 12:03:42,,,,,,,,



#### docker

docker 查看日志

docker container logs

docker exec -it  <id> bash

5a20ed4f947b

docker exec -it f9e79c4c1448 bash

 

#### 运行redis命令

```bash
docker run --name postredis -d -p 6379:6379 redis
```



#### 运行RocketMQ指令

```bash
# mq workspace
MQ_SPACE=$(dirname $(readlink -f "$0"))
echo MQ_SPACE:$MQ_SPACE
# make if absent
if [  ! -d $MQ_SPACE/namesrv/logs ]; then
  mkdir -p $MQ_SPACE/namesrv/logs
fi
# make if absent
if [  ! -d $MQ_SPACE/namesrv/store ]; then
  mkdir -p $MQ_SPACE/namesrv/store
fi
# run namesrv
docker run -d -p 9876:9876 -v $MQ_SPACE/namesrv/logs:/root/logs -v $MQ_SPACE/namesrv/store:/root/store --name rmqnamesrv -e "MAX_POSSIBLE_HEAP=100000000" rocketmqinc/rocketmq sh mqnamesrv

# make if absent
if [  ! -d $MQ_SPACE/broker/logs ]; then
  mkdir -p $MQ_SPACE/broker/logs
fi
# make if absent
if [  ! -d $MQ_SPACE/broker/store ]; then
  mkdir -p $MQ_SPACE/broker/store
fi
# run broker
docker run -d -p 10911:10911 -p 10909:10909 -v  $MQ_SPACE/broker/logs:/root/logs -v  $MQ_SPACE/broker/store:/root/store -v  $MQ_SPACE/broker.conf:/opt/rocketmq/conf/broker.conf --link rmqnamesrv:namesrv -e "NAMESRV_ADDR=211.71.76.189:9876" -e "MAX_POSSIBLE_HEAP=200000000" rocketmqinc/rocketmq sh mqbroker -c /opt/rocketmq/conf/broker.conf
# console
docker run -d --name rocketmq-dashboard -e "JAVA_OPTS=-Drocketmq.namesrv.addr=211.71.76.189:9876" -p 8088:8080 -t apacherocketmq/rocketmq-dashboard:latest
```



> 监控地址http://211.71.76.189:8088/#/topic



#### Hbase Docker

https://cloud.tencent.com/developer/article/1632053

```
docker run -d -h docker-hbase \
        -p 2181:2181 \
        -p 8011:8080 \
        -p 8085:8085 \
        -p 9090:9090 \
        -p 9000:9000 \
        -p 9095:9095 \
        -p 16000:16000 \
        -p 16010:16010 \
        -p 16201:16201 \
        -p 16301:16301 \
        -p 16020:16020\
        --name hbase \
        harisekhon/hbase:1.4
```



##### Hbase Dashboard

> http://211.71.76.189:16010/master-status



Hbase 表格

1 快递员数据表格	

```
create 'postman_test','id','post_arrive_info','day','region','post_id'
```

#### Mysql	

```bash
 docker run -p 3306:3306 --name post-mysql -v /data/MengQingqiang/postweb/mysql/data:/var/lib/mysql -e MYSQL_ROOT_PASSWORD=669988  -d mysql:latest
```



docker exec -it fc0f68f00544 bash

mongo pass 123456

### docker root

```
sudo docker exec -ti -u root 70ce555dbfb1da80716db5bb1539b730b1d5964ab196323b3903d8b2b01e79ad bash
```



#### 运行torchserve 命令

前台

```bash
docker run --rm -it -p 6080:8080 -p 6081:8081 -p 6082:8082 -p 7070:7070 -p 7071:7071 --volume /data/MengQingqiang/rpc/model_store:/tmp/models torchserve:v1.1 torchserve --start --model-store /tmp/models/ --models greedy_time.mar,greedy_distance.mar,pointer_net.mar --no-config-snapshots
```

后台

~~~
docker run  --rm -it -p 6080:8080 -p 6081:8081 -p 6082:8082 -p 7070:7070 -p 7071:7071 --volume /data/MengQingqiang/rpc/model_store:/tmp/models torchserer:v1.2.1 torchserve --start --model-store /tmp/models/ --models greedy_distance.mar --no-config-snapshots

sudo docker run -d --rm -it -p 6080:8080 -p 6081:8081 -p 6082:8082 -p 7070:7070 -p 7071:7071 --volume /data/MengQingqiang/rpc/model_store:/tmp/models pytorch/torchserve:0.5.0-cpu torchserve --start --model-store /tmp/models/ --no-config-snapshots

pytorch/torchserve
~~~

打包命令

```bash
torch-model-archiver --model-name greedy  --model-file ../serve/examples/image_classifier/greedy/greedy_distance.py --handler ../serve/examples/image_classifier/greedy/greedy_handler.py  --runtime python3 --version 1.0
```



0 不存在

1 已创建任务

2 预处理开始

3 预处理完成，开始生成数据

4 完成生成数据，开始写数据库

5 完成





```
1. nnictl experiment show        show the information of experiments
2. nnictl trial ls               list all of trial jobs
3. nnictl top                    monitor the status of running experiments
4. nnictl log stderr             show stderr log content
5. nnictl log stdout             show stdout log content
6. nnictl stop                   stop an experiment
7. nnictl trial kill             kill a trial job by id
8. nnictl --help                 get help information about nnictl
 sudo mount 10.126.56.9:/data/MengQingqiang/nni /data/MengQingqiang/mqq/nni

nnictl create --config egcn_con.yml --port 9074
```





### 恢复容器

```bash
# mysql: 
docker start fc0f68f00544
# rocketmq brocker
docker start b95beabd4365
# rocketmq namesrv
docker start 72120183816a
# dashboard
docker start 19329818936a
# mongodb
docker start f4b554eaafca
# redis 
docker start bfaa84ad7062
```

