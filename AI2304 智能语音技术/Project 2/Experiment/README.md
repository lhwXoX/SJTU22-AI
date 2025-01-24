# Project 2：LVCSR系统搭建

#### 修改
编写了`project.slurm`用于提交`run.sh`
在`run.sh`修改了`local/nnet3/run_tdnn.sh --num_jobs_final 1`以适配GPU
在`cmd.sh`中修改为

- `export train_cmd="run.pl --mem 16G"`
- `export decode_cmd="run.pl --mem 16G"`
- `export mkgraph_cmd="run.pl --mem 16G`

以满足对内存的需求

在path中将`export KALDI_ROOT="/lustre/home/acct-stu/stu1718/kaldi"`取消注释并注释`export KALDI_ROOT="/lustre/home/acct-stu/stu1718/kaldi_cpu"`

---

#### 使用方法
在命令行中输入`sbatch project.slurm`

---

#### 模型路径
`/lustre/home/acct-stu/stu1864/aishell/exp/nnet3/tdnn_sp/final.mdl`，可在[https://jbox.sjtu.edu.cn/l/M13DM5](https://jbox.sjtu.edu.cn/l/M13DM5)下载