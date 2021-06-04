# Seq2seq
seq2seq 机器翻译项目

### 项目简述
在于理解 Seq2seq 模型，代码来源于 https://github.com/keon/seq2seq。

### 理解要点：
1. Seq2seq 模型与注意力机制
2. Beam Search
3. Greedy Decoding

### 问题：
1. 训练和预测时，长度一致吗？

答：训练时：根据标签的长度进行 decoder 的解析。
   预测时：根据 max_length 进行解析，遇到 SOS 结束标签 break 解析。



