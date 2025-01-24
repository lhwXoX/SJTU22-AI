# Project 1：LLM基底模型的指令微调

---

以小组为单位，对Qwen-2.5-0.5B基底模型在Alpaca指令数据集上进行全量微调，并在一系列评测数据集上比较微调前后的模型性能。具体的大作业要求，请查阅在线文档：[飞书](https://vs74u9rj3b.feishu.cn/docx/TN4bdslXboBGjExf7XacpQHYnAc)

请同学们事先在人员页面的“大作业一”小组集中加入小组、组成1~3人的小组，并以小组形式提交作业。

---
- **微调后的Qwen-2.5-0.5B:** [model.safetensors](https://jbox.sjtu.edu.cn/l/bH1iaH) (由于kaggle训练时间的限制，我们只训练了30000steps)
- **训练脚本:** [finetune-eval.ipynb](./Experiment/finetune-eval.ipynb)
- **微调前模型评测结果:** [plm.csv](./Experiment/plm.csv)
- **微调后模型评测结果:** [sft.csv](./Experiment/sft.csv)
- **第一次大作业报告:** [Report.pdf](./Report/Report.pdf)