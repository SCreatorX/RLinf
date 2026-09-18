OpenWAM 监督微调
=================

本配方通过 RLinf 的 Ray 管理 FSDP runner，在原生 LIBERO 数据集上微调 OpenWAM。强化学习训练见 :doc:`openwam`。训练会从 OpenWAM checkpoint 目录读取模型配置，复用原生 dataloader 和视频、动作联合 loss，并使用 Full-shard FSDP 更新未冻结模块。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenWAM Wan2.2

   .. grid-item-card:: 方法
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: 数据
      :text-align: center

      原生 LIBERO reader

   .. grid-item-card:: 硬件
      :text-align: center

      2 张及以上 GPU 与 FSDP

OpenWAM checkpoint 会提供模型和 dataloader 设置。将 ``data.train_data_paths`` 指向数据集根目录（也可以是多个根目录的列表，各数据集用同一套 dataloader 设置读取后按样本数比例拼接）；loader 会读取 ``actor.model.model_path`` 下的 ``config.yaml``，并保留原生的帧数、动作和归一化约定。

安装
----

安装 OpenWAM 环境和 RLinf：

.. code:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

运行
----

在 ``examples/sft/config/model/openwam.yaml`` 和 ``examples/sft/config/libero_sft_openwam.yaml`` 中设置 checkpoint 与数据集路径。配方默认将 actor 放到 GPU ``0-1``，并使用 ``use_orig_params: true``，因为 OpenWAM 会冻结部分 backbone，同时训练 action 模块。

启动由 Ray 管理的 FSDP runner：

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

修改 GPU 数量时，同时修改 ``cluster.component_placement.actor``，并确保 ``actor.global_batch_size`` 能被 actor world size 整除。

预设以 fp32 加载权重（``precision: fp32``），优化器持有 fp32 主权重，FSDP 用 bf16 计算（``mixed_precision.param_dtype``）；若主权重是 bf16，``lr: 1e-6`` 下几乎所有更新都会被舍入掉。模型预设同时保持 ``load_to_device: false``：每个 rank 先在 CPU 上构建模型，FSDP 在包装时把各自的分片搬到 GPU，任何一张卡都不会先装下整个模型。评测配方则用 ``load_to_device: true`` 直接加载到 GPU。

验证与断点续训
--------------

设置 ``data.val_data_paths``（一个或多个数据集根目录，用同一套 dataloader 设置读取）和 ``runner.val_check_interval`` 后，会在验证集上平均 OpenWAM 的原生 loss，记录为 ``eval/loss``、``eval/loss_video`` 和 ``eval/loss_action``。``actor.eval_batch_size`` 是每个 rank 的验证 batch，``actor.eval_max_batches`` 可以限制大数据集上每个 rank 跑的验证 batch 数。LeRobot 风格的读取器按 split 选取 episode：验证默认读 ``val`` split，如果验证集是一个只有 train split 的独立数据集，请设置 ``data.openwam_val_split: train``（验证集为空时会在启动阶段直接报错）。

checkpoint 会把 dataloader、sampler（含 shuffle 的 epoch）和随机数状态与模型权重一起保存，因此 ``runner.resume_dir=<log_path>/<experiment_name>/checkpoints/global_step_<N>`` 会从下一个未见过的 batch 继续，而不是重头开始这一轮数据。

查看结果
--------

在 TensorBoard 中观察 ``train/loss``、``train/loss_video`` 和 ``train/loss_action``。RLinf 会将 FSDP 模型和 optimizer shards 写入 ``runner.logger.log_path/<experiment_name>/checkpoints/global_step_<N>/actor``。

评估
----

LIBERO 评估配方与本配方使用同一套 checkpoint 约定。按标准 LIBERO 数据训练的 OpenWAM checkpoint 输出绝对 EEF10 目标位姿；RLinf 会在每个环境 step 根据当前实际位姿计算目标差值，再发送 7D OSC 动作。

设置 ``MUJOCO_GL=egl`` 和 ``PYOPENGL_PLATFORM=egl`` 后，可以先运行双环境 32 步 smoke test。当前配方将 env worker 与 rollout worker 分到不同 GPU，避免 EGL 渲染和 OpenWAM 推理争用同一张卡：

.. code:: bash

   python evaluations/eval_embodied_agent.py \\
     --config-path libero --config-name libero_spatial_openwam_eval \\
     env.eval.max_steps_per_rollout_epoch=32 env.eval.max_episode_steps=32

如果 checkpoint 使用 native delta EEF10 动作训练，将 ``env.eval.openwam_action_representation`` 改为 ``native_delta_eef10``，并确保它与数据 metadata 和评估 checkpoint 一致。


其他视频编码器
--------------

部分旧 OpenWAM checkpoint 使用 ``vjepa2_1``、``flux_vae`` 或 ``wan_vae`` 这些旧名称，部署时 RLinf 会自动转换。如果编码器权重不在 checkpoint 内，可设置 ``rollout.model.encoder_model_path`` 指向本机编码器目录。RLinf 会使用临时配置加载，不会修改 checkpoint。
