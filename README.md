# rnnlm
Tensorflow RNN ptb
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/models/rnn/ptb 를 수정하여
한글 지원 및 기능 추가

## Network
[http://nmhkahn.github.io/assets/RNN-Reg/p1-dropout.png]

## Train
    python train.py --model medium --data_path data/nsmc --save_path ./save_dir/

<pre>
Epoch: 37 Learning rate: 0.001
0.003 perplexity: 116.447 speed: 4391 wps
0.103 perplexity: 115.841 speed: 4407 wps
0.202 perplexity: 115.161 speed: 4404 wps
0.302 perplexity: 114.512 speed: 4405 wps
0.402 perplexity: 114.239 speed: 4404 wps
0.502 perplexity: 114.149 speed: 4405 wps
0.602 perplexity: 113.848 speed: 4382 wps
0.701 perplexity: 113.548 speed: 4378 wps
0.801 perplexity: 113.168 speed: 4381 wps
0.901 perplexity: 112.790 speed: 4384 wps
Epoch: 37 Train Perplexity: 112.611
Epoch: 37 Valid Perplexity: 121.635
Epoch: 38 Learning rate: 0.001
0.003 perplexity: 115.892 speed: 4398 wps
0.103 perplexity: 116.537 speed: 4403 wps
0.202 perplexity: 115.219 speed: 4402 wps
0.302 perplexity: 114.639 speed: 4404 wps
0.402 perplexity: 114.379 speed: 4403 wps
0.502 perplexity: 114.230 speed: 4404 wps
0.602 perplexity: 113.877 speed: 4368 wps
0.701 perplexity: 113.553 speed: 4373 wps
0.801 perplexity: 113.204 speed: 4377 wps
0.901 perplexity: 112.826 speed: 4380 wps
Epoch: 38 Train Perplexity: 112.632
Epoch: 38 Valid Perplexity: 121.634
Epoch: 39 Learning rate: 0.001
0.003 perplexity: 109.798 speed: 4394 wps
0.103 perplexity: 116.322 speed: 4405 wps
0.202 perplexity: 115.303 speed: 4403 wps
0.302 perplexity: 114.612 speed: 4404 wps
0.402 perplexity: 114.483 speed: 4404 wps
0.502 perplexity: 114.265 speed: 4404 wps
0.602 perplexity: 113.918 speed: 4371 wps
0.701 perplexity: 113.587 speed: 4376 wps
0.801 perplexity: 113.227 speed: 4379 wps
0.901 perplexity: 112.826 speed: 4382 wps
Epoch: 39 Train Perplexity: 112.646
Epoch: 39 Valid Perplexity: 121.630
Test Perplexity: 120.863
</pre>
 
## 문장 확률값 구하기
    python interence.py --save_path ./save_dir --prob True --sent "나는 문장입니다."
    
## 다음 단어 예측하기
    python interence.py --save_path ./save_dir --sent "다음 문장은"
