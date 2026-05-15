# 任务描述
本次任务你需要参考Readme文件，跑通pi0.5模型的torch infer。
此外，你需要生成10组inputs，每组包含noise，进行infer，并将最终的outputs dump出来。

dump到outputs目录。
python环境已经为你准备好了，即uv，在.venv目录下。
pytorch 版本的ckpt也已经下载好了，在

整个推理链路应该包含 前处理-模型-后处理，即完整的policy infer。
这是为了后面的编译部署来dump数据进行精度对点。
默认的torch实现，精度是bf16。
除了dump bf16的，我需要你将模型转为fp16，通用进行dump。

最终你需要产出：
- dump出两份inputs、outputs数据，一份bf16，一份fp16。两份数据的inputs应该是相同的，或者仅有fp16和bf16的差异。
- 分析一下bf16，fp16 outputs的差异，分析这是否正常，形成一个分析文档。
- 一个python 脚本，用来跑fp16的推理，load ckpt-> load inputs -> infer -> check with dumped outputs。