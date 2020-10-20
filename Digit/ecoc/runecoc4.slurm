#!/bin/bash
#SBATCH -J ecoc100                   # 作业名为 test
#SBATCH -o ecoc100.%j.out               # 屏幕上的输出文件重定向到 test.out
#SBATCH --array=1-10
#SBATCH --error error.out
#SBATCH --cpus-per-task=16        # 单任务使用的 CPU 核心数为 4
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=z.gao@student.vu.nl
#SBATCH -t 10:00:00                # 任务运行的最长时间为 1 小时
##SBATCH --begin=05:30:00                # 任务运行的最长时间为 1 小时

# 输入要执行的命令，例如 ./hello 或 python test.py 等
python digits100.py                    # 执行命令