
- DSTP-RNN: a dual-stage two-phase attention-based recurrent neural networks for long-term and multivariate time series prediction  
针对long-term的序列预测
指出前面的DA-RNN方法、GeoMAN方法的局限性。DA-RNN用于单步预测，GeoMAN用于短期预测，都难以捕捉长时依赖问题。
提出two phase的attention机制： The first phase produces violent but decentralized attention weight,
while the second phase leads to stationary and concentrated attention weight. 
和DA-RNN的不同之处：1、spatial-attention阶段分为了two-phase 2、spatial-attention阶段，first-phase不用target series信息，second-phase才使用target series信息。
从实验结果来看：提出的DSTP-RNN方法确实在捕捉长时依赖上相比DA-RNN和GeoMAN更胜一筹。


