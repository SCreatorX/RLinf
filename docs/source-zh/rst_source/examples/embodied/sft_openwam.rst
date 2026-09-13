OpenWAM 监督微调
=================

本配方通过 RLinf 的 Ray 管理 FSDP runner，在原生 LIBERO 数据集上微调 OpenWAM。训练会从 OpenWAM checkpoint 目录读取模型配置，复用原生 dataloader 和视频、动作联合 loss，并使用 Full-shard FSDP 更新未冻结模块。

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

OpenWAM checkpoint 会提供模型和 dataloader 设置。将 ``data.train_data_paths`` 指向数据集根目录；loader 会读取 ``actor.model.model_path`` 下的 ``config.yaml``，并保留原生的帧数、动作和归一化约定。

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

查看结果
--------

在 TensorBoard 中观察 ``train/loss``、``train/loss_video`` 和 ``train/loss_action``。RLinf 会将 FSDP 模型和 optimizer shards 写入 ``runner.logger.log_path/<experiment_name>/checkpoints/global_step_<N>/actor``。
