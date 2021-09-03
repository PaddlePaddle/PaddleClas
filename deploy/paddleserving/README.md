# Imagenet Pipeline WebService

This document will takes Imagenet service as an example to introduce how to use Pipeline WebService.

## Get model
```
sh get_model.sh
```

## Start server

```
python resnet50_web_service.py &>log.txt &
```

## RPC test
```
python pipeline_rpc_client.py
```
