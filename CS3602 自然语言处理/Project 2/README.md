# Project 2：构建聊天机器人

---

以小组为单位，在第一次大作业的基础上构建一个聊天机器人，并分析讨论聊天机器人的能力与表现。具体的大作业要求，请查阅在线文档：[飞书](https://vs74u9rj3b.feishu.cn/docx/ReWvdQn7WoWePxxvcFDcpe3nnWf)

请同学们事先在人员页面的“大作业一”小组集中加入小组、组成1~3人的小组，并以小组形式提交作业。

----
## NLP大作业第一部分
* **微调后的Qwen-2.5-0.5B:** [model.safetensors](https://jbox.sjtu.edu.cn/l/bH1iaH) (由于kaggle训练时间的限制，我们只训练了30000steps)
* **训练脚本:** [finetune-eval.ipynb](./Experiment/finetune-eval.ipynb)
* **微调前模型评测结果:** [plm.csv](./Experiment/plm.csv)
* **微调后模型评测结果:** [sft.csv](./Experiment/sft.csv)
* **第一次大作业报告:** [Project 1 Report.pdf](./Report/Project%201%20Report.pdf)
----
## NLP大作业第二部分
* **微调后的Qwen-2.5-1.5B-LoRA:** 微调过的两个Qwen模型(r=8, r=16)，可以在[Qwen-LoRA](https://jbox.sjtu.edu.cn/l/71i9fF)下载
* **Qwen基底模型，Instruct模型:** 可以在官方[Kaggle](https://www.kaggle.com/models/qwen-lm/qwen2.5)下载
* **Qwen-0.5B微调模型:** 将大作业一中的微调模型进行整合，可以在[Qwen2.5-0.5B-finetuned](https://jbox.sjtu.edu.cn/l/A12axl)下载
* **bert-base-cased:** 可以在[Hugging Face](https://huggingface.co/google-bert/bert-base-cased)下载
* **alpaca-cleaned:** 指令数据集可以在[HF Mirror](https://hf-mirror.com/datasets/yahma/alpaca-cleaned)下载
* **微调脚本:** [finetune.ipynb](./Experiment/finetune.ipynb)，使用LoRA实现对Qwen2.5-1.5B模型进行微调
* **聊天机器人代码:** [chat.py](./Experiment/chat.py), 基于Qwen-LoRA的聊天机器人；[chat_0.5_instruct.py](./Experiment/chat_0.5_instruct.py),基于大作业一中的Qwen-2.5-0.5B微调模型和Qwen-2.5-1.5B-Instruct的聊天机器人；[chat_base.py](./Experiment/chat_base.py)，基于Qwen-2.5-1.5B基底模型的聊天机器人，基底模型目前仅支持单句聊天
* **外部知识库:** [knowledge_base.txt](./Experiment/knowledge_base.txt)，在报告中使用的外部知识库
* **最终报告:** [Report.pdf](./Report/Final%20Report.pdf)
----
### 使用聊天机器人
#### Step 0: 创建环境
1. 首先在Anaconda中创建环境，测试所使用的python版本为3.12.3
```
conda create --n chatbot python=3.12.3
conda activate chatbot
```
2. 下载`pytorch`，可以在官网[PyTorch](https://pytorch.org/get-started/locally/)选择合适的版本，例如
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
3. 下载`transformers`, `peft`, `sentence_transformers`库
```
pip install transformers
pip install peft
pip install sentence_transformers
```
#### Step 1: 下载模型
1. 将预训练BERT模型下载至目录中的`bert-base-cased`文件夹中
2. 将基底模型Qwen-2.5-1.5B下载至目录中的`Qwen2.5-1.5B`文件夹中
3. 将微调模型Qwen-LoRA下载至目录`output-r=8`或`output-r=16`文件夹中(选择一个或全部下载)
4. 如果你想使用Instruct模型或微调的0.5B模型，请将模型分别下载至`Qwen2.5-1.5B-instruct`和`Qwen2.5-0.5B-finetuned`中

#### Step 2: 运行代码
1. 如果你想使用LoRA微调的模型，请先在`chat.py`的第205行选择你想使用的LoRA模型(r=8或16)，然后运行
```
python chat.py
```
2. 如果你想使用基底模型，请运行
```
python chat_base.py
```
3. 如果你想使用0.5B或Instruct模型，请先修改`chat_0.5_instruct.py`中的第199行选择使用0.5B还是Instruct模型，然后运行
```
python chat_0.5_instruct.py
```

#### Step 3: 聊天机器人设置
在运行代码后，你可以随时输入`\quit`结束对话或`\newsession`开启新的对话
一开始，你需要对聊天机器人进行初始化设置，请按顺序进行以下的设置
- **是否使用外部知识增强？(Yes/No)** 输入Yes或No表示是否从`knowledge_base.txt`中提取外部知识
- **检索几条外部知识？** 如果你启用了外部知识增强，可以在这里设置检索的知识条数
- **是否给予虚拟人身份？(Yes/No)** 输入Yes或No表示是否为聊天机器人赋予职业人设
- **请输入虚拟人身份(例如: a king in the kingdom):** 如果你启用了虚拟人设，可以在这里输入你想赋予的人设
- **是否输入说话人？(Yes/No)** 输入Yes或No表示是否将多个用户区分，让机器人针对不同用户进行输出(该功能目前性能不佳，后续工作将对其进行修改)，如果你选择了Yes，你需要在每轮对话开始时声明用户的身份
- **机器人以什么口吻聊天(例: humorous, 不需要请输入No)** 输入你想让机器人用什么口吻进行回复，如用默认回复请输入No
请注意，在使用基底模型进行输出时，由于基底模型的不稳定，暂时无法使用以上功能，进行的设定在聊天中无效

#### Step 4: 与聊天机器人尽情地聊天吧~