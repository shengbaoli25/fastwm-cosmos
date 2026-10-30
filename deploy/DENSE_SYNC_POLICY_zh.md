# DINOv3 / V-JEPA 同步推理适配器

`realman_sync.dense_policy.DensePolicy` 调用当前分支原有的
`DenseTeacher.encode_video(..., context_only=True)` 和
`DenseFeatureJointModel.infer_joint(...)`。训练入口、训练配置、模型实现、
损失函数、数据缓存和 checkpoint 格式均不修改。DINO 分支使用其原有 H+/16
图像编码和两帧特征平均；V-JEPA 分支使用其原有的两帧 tubelet 编码。

## 服务端需要的文件

复制并修改本分支对应的 `dense_policy.dinov3.example.json` 或
`dense_policy.vjepa21.example.json`。相对路径一律相对于部署 JSON 所在目录。

- `--checkpoint`：训练的 `training/checkpoints/weights/step_XXXXXX.pt`。
  不要传 DeepSpeed 优化器状态目录。
- `training_config`：**该次训练保存的 `training/config.yaml`**，不能直接传
  `configs/experiments/` 下尚未绑定训练身份的模板。适配器核对配置与 checkpoint
  的完整 manifest、模型尺寸、teacher、缓存身份和时序语义，并严格加载所有权重。
  此配置也提供 checkpoint manifest 尚未记录的 `force_shift`，不静默猜测。
- `normalization_file`：**该次 run 根目录的 `normalization.json`**，由
  `scripts/dense/runtime.py` 保存，包含仅训练 episode 的统计量。不要替换成
  对所有 episode 重新统计或旧 Cosmos 导出的 normalization 文件。
- `teacher_source`、`teacher_weights`：训练所用的官方冻结编码器源代码和权重。
  调用原有 loader 核对 checkpoint 中的 SHA256 和固定 upstream revision；
  不在服务启动时自动下载。冻结 teacher 不包含在训练模型 checkpoint 中。
- `force_preprocessing_file`：训练数据的 `meta/force.json`，读取其 `model_input`。
  也可用 `force_preprocessing` 直接嵌入该 contract。当前训练保存的 normalization
  文件不包含此字段，所以需要单独提供。客户端必须使用同一标定、滤波与坐标约定。

DINOv3 需要按原项目要求接受其许可证后设置 `DINOV3_LICENSE_ACCEPTED=1`。
服务端使用原分支已有训练/推理 Python 环境，且要初始化其原有模型依赖；
机器人 Client 不需要安装 PyTorch、teacher 或模型权重。模型和 teacher 在启动时
各加载一次。CPU 是显式测试选项；完整模型的实际部署应配置可容纳其权重的 GPU。

```bash
# 在对应 DINO 或 V-JEPA 分支仓库根目录运行。替换配置和 checkpoint 路径。
python deploy/serve_policy.py --backend dense \
  --config deploy/dense_policy.dinov3.example.json \
  --checkpoint /absolute/dinov3-run/training/checkpoints/weights/step_020000.pt \
  --host 0.0.0.0 --port 8766
```

## 在线输入与动作

两路相机按照 `camera_order=[global,wrist]` 对应训练的 main/wrist 顺序。
客户端发送 RGB24 原始字节和原始尺寸；服务端按训练数据路径先用 bilinear
antialias 缩放至每路 224×224、水平拼接，再交给原 teacher 进行 ImageNet
归一化及其编码处理，不把压缩图像字节误当 RGB。

两个分支都需要 **30 Hz 采集网格上的 `[-4,0]` 两帧**，约相隔 133.33 ms。
`/health` 返回 `video_context_frames=2, video_context_stride=4`。客户端保持
至少五个实际采集点，选择第 -4 帧和当前帧，不能使用上一轮 HTTP 请求的图像
代替。模型没有读取未来帧、未来动作或未来力标签。

数值历史是最近 32 个采集点的 `[joint7_rad, gripper_open, force6]`，最后一点
必须和当前 observation 完全匹配；真实客户端启动时等待历史满 32 点。适配器
仍支持训练相同的左侧补零和有效性 mask。所有时间比较使用客户端的 monotonic
时钟，服务端不把客户端时间戳与服务器时钟相减。

推理结果先按训练 FP32 scale/offset 反归一化，再返回绝对关节目标（7 维 rad）
和夹爪 openness（1 维，截断至 [0,1]）。不是关节增量，也不是末端 pose。
`action_horizon` 只能在 1..32，返回完整预测的前 N 步；同步 Client 配置决定
本轮实际执行前多少步。机械臂关节限制和动作执行留在 Client。

## 验证范围

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src:deploy OMP_NUM_THREADS=2 \
  python -m pytest -q deploy/tests/test_sync_dense_policy.py
```

测试覆盖路径解析、训练配置/checkpoint/归一化身份不匹配拒绝、严格权重形状检查、
相机顺序与像素预处理、[-4,0] 时序、history/proprio 形状、实际动作反归一化，
以及原有小尺寸 DenseFeatureJointModel 的一次真实 CPU flow 推理。
冻结 teacher 使用小型替身，因此这些测试不代表完整官方权重的 GPU 性能或真实
机械臂任务成功率；正式启动时仍需实际 run 产物和官方编码器。
