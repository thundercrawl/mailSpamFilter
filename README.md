
[TODOs]
1. support N-GRAM  word-embedding
2. Q&A model support

[Support features.]

1. tensorflow spam filter based on NN
2. support chinese/english
3. support multi training
4. provide tensorflow model server and consume by grpc


[Steps]

1.train, start worker and ps
training start in three different command windows

python3 train.py --job_name=ps --task_index=0
python3 train.py --job_name=worker --task_index=0
python3 train.py --job_name=worker --task_index=

output like:
step 12480, change in cost 0.00835037
step 12490, training accuracy 0.96206
step 12490, cost 20.9695
step 12490, change in cost 0.0092144
WARNING:tensorflow:From train.py:214: The name tf.train.Saver is deprecated. Please use tf.compat.v1.train.Saver instead.

final accuracy on test set: 0.85
close training,exit


2.predict in local files
python3 predict.py  --mail /opt/ai/data/dataset/spam/y31.txt 

3.predict use the tensorflow grpc server
  --start model server to load the model in tranning section
  tensorflow_model_server  --model_name=mlmodel --model_base_path=/opt/ai/mail/tensorflow-tutorial/model

  server will listen on default 8500 port
    2019-08-29 13:44:29.668960: I tensorflow_serving/model_servers/server.cc:324] Running gRPC ModelServer at 0.0.0.0:8500 ...
  --consume the saved model by grpc
  python3 grpcClient.py --mail /opt/ai/data/dataset/spam/d26.txt 
  output like:
  mail is: dtype: DT_FLOAT
tensor_shape {
  dim {
    size: 1
  }
  dim {
    size: 2
  }
}
float_val: 0.5244612693786621
float_val: 0.4781316816806793

4. mlflow support
copy the whole directory to any production folder, and run following command

mlflow run mlflow run ./mailFilter/ -P multi=2 -P numEpochs=100

mlflow models serve --model-uri runs:/240da73665424e15b0e30e6bd8ca0d80/mlmodel --port 51000
