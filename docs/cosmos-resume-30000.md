# Cosmos 25000 → 累计 30000 的恢复入口方案

本独立分支 `codex/cosmos-rm71d78-resume30k` 从 `057f787` 创建，不包含三个新基线分支的改动。
原 Cosmos 模型、数据和训练源码保持不变；本次只新增 `scripts/cosmos_resume/` 与本说明。

状态：[resume_to_step.py](../scripts/cosmos_resume/resume_to_step.py) 与
[launch_platform.sh](../scripts/cosmos_resume/launch_platform.sh) 已实现本方案，默认plan。
真实GPU恢复尚未启动；执行依赖远端审计文件。4项wrapper测试与4项audit测试已通过。
其中真实Accelerate1.12.0 CPU分片测试覆盖原64462长度、四rank B64、恢复剩余200个batch、
epoch末padding、下一epoch顺序，以及重建iterator不推进model RNG；独立只读审阅未发现确定阻断。

## 已确定的配置

本机 `C:/Users/ASUS/Downloads/config.yaml` 是 2313 字节，output_dir 精确匹配原任务
`job-187fc8ee-5443-4786-963b-d48882d3f8ff/runs/long/training`。
已原样复制为 [original-config.yaml](../scripts/cosmos_resume/original-config.yaml)，SHA256：
`6941b393f453a57d42942a70e0780853405bbdeb37e38fe3b6d4393deb6e6898`。
远端执行 `sha256sum "$OLD/config.yaml"` 即可核对是否字节一致，不必再次下载。

原任务 Python（来自平台启动命令）：
`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python`。
平台已确认原任务停止，使用 4×H200，每卡 batch64。

原配置必须保持：batch64、workers8、accum1、BF16、seed42、AdamW lr1e-4、weight decay0.01、
cosine、clip1.0、save5000、eval0、max_steps50000，以及 model.training_recipe 中的
`batch_per_rank=64,max_steps=50000`。model factory 是
`fastwam.cosmos.assets.create_noaction_joint_cosmos`，数据为原 job 的 CosmosCachedDataset。

- features：原 job 下 `prepared/features`，identity
  `13d25abda35ecb39e1d2efb52239ae2438993cb8b693a4c533f5465bfca20224`。
- text：原 job 下 `prepared/text`，identity
  `a73a28af6561922cfca270c7e67496ca15e186df244c392e69bb2da0f2bc0917`。
- normalization identity：`ac2a30c48f9a5d8f92803d05af861bfde2f72872eb473d83cb27f5f353de1236`。
- upstream：`/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-noaction-joint8/code-82cb9d2/third_party/cosmos-predict2.5`。
- base checkpoint：同 `realman-noaction-joint8/assets/base/base/post-trained/81edfebe-bd6a-4039-8c1d-737df1a790bf_ema_bf16.pt`。
- action/force/memory seeds42/43/44；AE768/FFN3072，Force768/FFN2048/heads16；joint8；fps7.5；gradient checkpointing开启。

## wrapper 结构

已新建独立 `resume_to_step.py`，使用新的 job 输出目录，不修改或覆盖原 run、缓存、源码和 state。
默认 plan；只有显式 `--execute` 且审计条件满足才进入 GPU worker。
原代码根为任务共享 `cosmos/code`，但必须先核查其 HEAD、修改和关键文件哈希与原运行版本的一致性。

1. 入口解析 `--config`、`--state`、`--output`、`--source-root`、`--stop-at-step 30000`。
   核对配置 SHA256、原 job/output身份、完整 state 审计、源码清单；输出目录须新建。
   不调用旧 dd805094 resume 控制器或 fresh-training preflight 配方验证。
2. 使用原 Python 和原四 rank ZeRO1配置。设置 `PYTHONPATH=<verified source>/src:<verified source>`，
   `CUDA_VISIBLE_DEVICES=0,1,2,3`；平台分布式父环境中的旧 rank变量须按原启动器方式处理，
   防止嵌套启动产生8/16进程。wrapper内部核对rank0..3、world4、实际四个H200。
3. `torch.cuda.set_device(LOCAL_RANK)`，创建 `Accelerator(gradient_accumulation_steps=1,
   mixed_precision='bf16', step_scheduler_with_optimizer=False)`；seed42。
4. 加载**原配置**，仅设置新 `cfg.output_dir` 与完整目录 `cfg.resume`。
   保持 `cfg.max_steps=50000`、`cfg.model.training_recipe.max_steps=50000`。
   按原 Cosmos 入口 `instantiate(cfg.model, device=...)`，随后创建原 cached dataset，
   val与train为同一对象，长度必须64462。不要走通用 `runtime.run_training`，该入口额外传递
   model_dtype；应沿用原 Cosmos 入口的工厂调用签名。
5. 构造 `Wan22Trainer(model, train_ds, val_ds, cfg=cfg)`；其构造器以
   `accelerator.load_state(state_dir)` 完整恢复。核对原 trainer源码确实如此。
6. 每rank在任何新 optimizer update 前，核对：
   - trainer progress精确等于trainer_state.json；engine.global_steps等于25000。
   - 引擎实际 microbatch64、accum1、globalbatch256、ZeRO1、BF16。
   - Adam所有已建状态step为25000、master weights与moments有限；model权重有限。
   - scheduler.state_dict与保存的scheduler.bin一致；原50k方案应为warmup2500、
     cosine47500、SequentialLR last_epoch25000。以实际原源码和保存state为准。
   - Python/NumPy/Torch/CUDA RNG、Accelerate内部step与各rank random_states_N.pkl一致。
   - ForceRuntimeState的manifest_bytes与rng_state与保存模型一致；原严格prehook必须保留。
   这些是真实恢复阶段的检查，会读取state；只读审计脚本不替代它们。
7. 处理数据进度：旧原生sampler跳过raw samples，可能改变本epoch最后一个padding batch。
   若远端仍是本地版本，改在wrapper使用已有 `FirstEpochContinuation` 的思路：清除原生offset，
   使用 `accelerate.skip_first_batches` 跳过**已分片的batch**，恢复原epoch；创建worker iterator时
   保存/恢复model RNG。缓存数据确定、无随机图像增强。不可照搬旧verify_rng_and_sampler中的B16硬编码。
   对dataset64462、globalbatch256，完整epoch有252次更新；step25000对应epoch99/batch52只是推断，
   实际offset始终从trainer_state.json读取。
8. 全部rank恢复检查成功后，**仅设置 `trainer.max_steps=30000`**，保存新的resume receipt：
   原配置hash、原50k scheduler/recipe、原state路径与step、stop_at_step30000、新job、各rank实际参数。
   config.yaml保留原50k训练语义和新的output/resume；单独receipt记录提前停止目标，避免改坏manifest。
9. `trainer.train()`；确认所有rank结束step30000，原生最终完整保存完成，有权重及全state，
   再核对最终checkpoint结构与日志。保存5000间隔下30000会先周期保存、再最终保存；
   需要最终完成记录，不能只见周期保存就认定作业完成。

## 平台文件与命令

远端先获取这个独立分支，使用一个新的目录，不覆盖原 `$COSMOS/code`：

```bash
git clone --single-branch --branch codex/cosmos-rm71d78-resume30k \
  https://github.com/shengbaoli25/fastwm-cosmos.git /work/cosmos-resume30k-kit
git -C /work/cosmos-resume30k-kit rev-parse HEAD
```

将 `audit_resume.py`、`resume_to_step.py`、`launch_platform.sh` 放入
`/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-cosmos-50k-20261029/cosmos/resume30k/tools/`。
它们位于新clone的 `scripts/cosmos_resume/`。原模型运行源码仍使用 `$COSMOS/code`，
这个新clone只提供恢复工具。先运行 `bash -n launch_platform.sh` 做平台shell语法核验。
先在原Python环境执行以下审计；JSON写入新的tools目录，不触碰旧训练目录：

```bash
COSMOS=/inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-cosmos-50k-20261029/cosmos
PY=/inspire/hdd/project/long-range-embodied-collaborative-intelligence/ky26340/realman-fastwam/.venv/bin/python
OLD="$COSMOS/online/job-187fc8ee-5443-4786-963b-d48882d3f8ff/runs/long/training"
"$PY" "$COSMOS/resume30k/tools/audit_resume.py" "$OLD" \
  --source-root "$COSMOS/code" --world-size 4 --stop-at-step 30000 --archive-index \
  > "$COSMOS/resume30k/tools/remote-audit.json"
bash "$COSMOS/resume30k/tools/launch_platform.sh" --plan
```

审查remote-audit.json并确认源版本后，新建4×H200训练作业，平台启动命令为：

```bash
bash /inspire/ssd/project/long-range-embodied-collaborative-intelligence/ky26340/rm71d-jepa-cosmos-50k-20261029/cosmos/resume30k/tools/launch_platform.sh --execute
```

脚本自动使用新 `TRAIN_JOB_ID` 创建 `cosmos/resume30k/online/<new-job>/training`，
保留本地训练日志与四rank恢复receipt。没有`--execute`只打印计划。
wrapper核对audit中的源码哈希、配置SHA和state文件大小/修改时间，再实际加载完整state；
原来的没有sampler.bin不会阻断恢复，epoch/batch来自trainer_state.json。
原JSON没有gradient_clipping字段，配置max_grad_norm1不证明DeepSpeed引擎实际clip1；
本wrapper保留原引擎配置，并在receipt记录实际值，不添加新裁剪行为。

## 只读审计与本地验证

`audit_resume.py` 仅检查显式传入的oldrun与可选source-root；默认只读取文件元数据和小型文本，
不导入Torch、不调用torch.load、不反序列化pickle、不启动训练。`--archive-index`仅读取ZIP中央目录，
不读取tensor内容，不代表文件的全部校验和或数值正确。未出现sampler.bin仅报告warning。
审计产物包含原config、源码哈希、完整state文件结构和候选状态；`ready_to_resume`始终为false，
实际恢复和逐rank验证由显式执行的wrapper完成。

在安装Torch、Accelerate1.12.0、OmegaConf和NumPy的环境中，从repo根目录运行：

```bash
python -m unittest discover -s scripts/cosmos_resume -p 'test_*.py' -v
python -m compileall -q scripts/cosmos_resume
```

验证证据位于 [validation.json](../scripts/cosmos_resume/validation.json)。
完整模型/优化器的GPU恢复、实际到30000步训练和最终checkpoint尚未在本次本地验证中执行。

## 尚缺的远端证据

- 配置SHA256匹配；原 `cosmos/code` 与关键原源码/运行版本一致。
- `state/step_025000` 全四rank文件实际存在、非空且可读；没有更新的完整state。
- features/text/base checkpoint/upstream仍存在、身份匹配；原环境依赖版本与启动配置。
- 实际saved scheduler/trainer_state，及远端sampler/loader实现，用于确定恢复代码。

没有这些证据之前，不生成声称已验证的执行命令、不启动训练。
