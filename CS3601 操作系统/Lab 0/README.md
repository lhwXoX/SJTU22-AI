# Lab1-Bomb

个人答案见：[ans.txt](./ans.txt)
任务描述见：[lab-instructions.pdf](./lab-instructions.pdf)
请参考实验环境搭建文档[env.pdf](./env.pdf) ：获取关于搭建实验环境、完成实验以及提交实验结果的指南。

---

This is the repository of the bomb lab in SJTU CS3601.

An online tutorial: [https://www.bilibili.com/video/BV1q94y1a7BF/?vd_source=63231f40c83c4d292b2a881fda478960]

## Get Your Bomb
**Please enter your student number CORRECTLY in student-number.txt at first!!!**

**Otherwise, you would receive 0 points in this lab.**

- `make bomb`: get your personal bomb from bomb warehouse

## Emulate

- `make qemu`: Start a QEMU instance to run your bomb

## Debug with GDB

- Start a terminal and run `make qemu-gdb`: Start a QEMU instance with GDB server
- Start another terminal and run `make gdb`: Start a GDB (gdb-multiarch) client

## Other

- Press `Ctrl+a x` to quit QEMU
- Press `Ctrl+d` to quit GDB
- Read `doc.pdf` for more reference
