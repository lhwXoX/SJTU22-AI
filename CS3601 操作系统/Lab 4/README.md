# Lab 4：多核调度与IPC

实验报告见：[Report.pdf](./Report/Report.pdf)
实验代码见：[Experiment](./Experiment)

---

## 【实验环境】
本实验采用与拆弹实验相同的实验环境，请参考[env.pdf](../Lab%200/env.pdf)

若遇到docker镜像版本问题，请参考[关于Docker](https://sjtu-ipads.github.io/OS-Course-Lab/Getting-started.html#%E7%8E%AF%E5%A2%83%E5%87%86%E5%A4%87)更新镜像

## 【实验代码】
本实验代码发布在 Github 上，提交则直接使用 Canvas。

在你的实验环境的命令行中执行以下命令克隆实验仓库（在拆弹实验中已完成）：
```
$ git clone https://github.com/SJTU-IPADS/OS-Course-Lab.git
```
克隆完成后，执行以下命令切换到main分支的Lab4实验目录，后续操作都在此目录中进行：
```
$ cd OS-Course-Lab
$ git checkout main
$ git pull
$ ./Scripts/gendeps.sh
$ cd Lab4
```
请参考[在线指导网页](https://sjtu-ipads.github.io/OS-Course-Lab/)完成实验。

请将实验报告放在代码仓库中（即 OS-Course-Lab/Lab4 目录下），格式不限。

## 【实验提交】
完成实验后，执行下列命令来保存你的更改。
```
$ git add -A
$ git commit -m "finish ChCore-Lab4"
```
然后，执行下列命令打包你的实验代码及实验报告，将<实验报告路径>替换成你的实验报告路径，并将压缩包chcore-lab4.tar.gz提交到 Canvas 作业中即可。
```
$ make submit
$ tar -zcf chcore-lab4.tar.gz lab4.tar.gz <实验报告路径>
```

## 【注意事项】
- 请确保正确执行了`git pull`以将仓库更新为最新版本
- 当遇到编译报错时, 尝试先执行`make clean`再编译
- 请仔细阅读[在线指导网页Lab 4](https://sjtu-ipads.github.io/OS-Course-Lab/Lab4.html)中的内容
- 在mac上运行实验环境的同学需要更新`jasonsjtu/oslab`镜像，否则会出现"ead_procmgr_elf_tool: not found"报错