OpenWAM 模型强化学习训练
========================

本示例把 OpenWAM 视频世界模型接入 RLinf 的 PPO 流程：HuggingFace rollout worker 用原生联合推理生成动作并缓存去噪链，FSDP actor 精确重放该链计算可微的 log-probability 与 value，训练结束后可导出为 OpenWAM 原生 checkpoint 目录用于部署和评测。SFT 见 :doc:`sft_openwam`。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      OpenWAM（Wan2.2 双系统，其他架构见下表）

   .. grid-item-card:: 方法
      :text-align: center

      PPO（去噪转移代理似然）

   .. grid-item-card:: 环境
      :text-align: center

      LIBERO spatial / object / goal / 10

   .. grid-item-card:: 硬件
      :text-align: center

      8 张 GPU：2 张 actor、1 张 rollout、其余给渲染

设计要点
--------

- **执行前缀而不是整块动作。** 模型每次预测 ``num_frames - 1 = 32`` 步动作，OpenWAM 的部署执行器只执行前 ``inference_horizon = 10`` 步就重新生成。整块 32 步开环执行在 LIBERO-Spatial 上成功率为 0，执行 10 步约 80 步完成任务。``rollout.model`` 与 ``actor.model`` 都要设置 ``openwam.inference_horizon: 10`` 和 ``num_action_chunks: 10``；PPO 的行为与 actor log-probability 只覆盖被执行的 10 帧。
- **动作表示要和 checkpoint 一致。** ``new-openwam-libero-sft-10epoch-delta-aligned`` 输出 OSC 单位的 EEF10 增量，用 ``openwam_action_representation: native_delta_eef10``；旧的 ``openwam-libero-sft-30000`` 输出绝对末端位姿，需要 ``absolute_eef10``。
- **可微重放。** rollout 在 ``predict_action_batch`` 中以高斯探索采样一个去噪转移，并把 OpenWAM 非扁平的原生条件（Cosmos ``und_kv``、tri-system 的 VLM 特征等）用 ``rlinf.models.embodiment.openwam.replay.pack_native_inputs`` 打包成定形张量；actor 侧 ``default_forward`` 逐样本解包重放，未更新权重时 ``actor/ratio`` 精确为 1.0、``actor/approx_kl`` 为 0。
- **value head。** ``actor.model.add_value_head: true`` 时 RLinf 在视频 latent、文本 context、本体感知和动作四组池化统计量上接一个小 MLP。它随 FSDP checkpoint 保存，导出时写入 ``rlinf_value_head.pt``，从导出目录续训时会自动读回。

支持的架构
----------

PPO 桥接按 checkpoint 的架构分派，任何 ``forward(noisy_actions, action_timestep, proprio=..., **pipeline_inputs)`` 返回动作 flow 预测的 OpenWAM 架构都可以训练。``toolkits/openwam/check_ppo_replay.py`` 对单个 checkpoint 做 rollout → 轨迹传输 → actor 重打分 → 反向传播的模型级校验，并把指标写成 JSON。

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - 架构 / 主干
     - 状态
   * - dual_system ``joint_self_attn``（Wan2.2 TI2V 5B）
     - 已完整验证，所有 LIBERO PPO 运行使用
   * - dual_system ``joint_cross_attn``、``idm``
     - 模型级校验通过
   * - single_system ``vanilla``、``moe``
     - 模型级校验通过
   * - tri_system ``joint_self_attn``（Qwen3-VL）
     - 模型级校验通过；VLM 必须冻结，其隐状态在每个观测上计算一次并随轨迹重放
   * - Wan2.1 VACE 1.3B、Wan2.1 I2V 14B
     - 模型级校验通过；14B 单样本重放峰值约 117 GB，需跨 actor rank 使用 FSDP
   * - Cosmos3 Edge
     - 模型级校验通过；``unify_action`` 把 20 维物理状态散射到 80 维统一向量
   * - Cosmos-Predict2.5
     - 未测试，需要 OpenWAM 的 ``install_cosmos_predict25.sh`` 附加依赖

安装
----

.. include:: _setup_common.rst

.. code-block:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

LIBERO 的 EGL 渲染需要 GLVND 的 ``libEGL.so.1``（Debian/Ubuntu 上为 ``apt-get install libegl1``），并设置 ``MUJOCO_GL=egl``、``PYOPENGL_PLATFORM=egl``。

运行
----

配置文件
~~~~~~~~

- ``examples/embodiment/config/libero_spatial_ppo_openwam.yaml``：完整配方（64 环境、8 个 rollout epoch）。
- ``examples/embodiment/config/libero_spatial_ppo_openwam_long.yaml``：持续验证配方（8 环境、320 步 episode、4 步去噪、20 轮）。
- ``examples/embodiment/config/libero_spatial_ppo_openwam_smoke.yaml`` 与 ``tests/e2e_tests/embodied/libero_spatial_ppo_openwam.yaml``：冒烟配方。
- ``examples/embodiment/config/robotwin_click_bell_ppo_openwam.yaml``：RoboTwin click_bell 上的 PPO（aloha-agilex，20 维绝对 EEF，整块 32 步，每轮 4 环境 52 个样本），环境适配见 :doc:`../../evaluations/guides/robotwin`；尚未在真实 RoboTwin 环境里跑过。

关键片段：

.. code-block:: yaml

   env:
     train:
       openwam_action_representation: native_delta_eef10
       total_num_envs: 4
       render_gpu_ids: [4, 5, 6, 7]   # 每个 LIBERO 子进程一张渲染卡，且不能与 CUDA 计算共卡
       task_id_filter: [2, 3, 4]      # 可选：只在指定任务上训练

   rollout:
     model:
       model_type: openwam
       model_path: /path/to/new-openwam-libero-sft-10epoch-delta-aligned
       num_frames: 33
       denoise_steps: 4
       num_action_chunks: 10
       openwam:
         inference_horizon: 10

   actor:
     model:
       model_path: ${rollout.model.model_path}
       add_value_head: true
       openwam:
         rl_enabled: true
         noise_std: 0.05
         inference_horizon: 10
     micro_batch_size: 1
     global_batch_size: 128

批大小约束
~~~~~~~~~~

每轮采集的动作块样本数为 ``total_num_envs * rollout_epoch * (max_steps_per_rollout_epoch / num_action_chunks)``，必须能被 ``actor.global_batch_size`` 整除，后者又必须能被 ``actor.micro_batch_size * actor_world_size`` 整除。RLinf 在加载模型之前校验这些约束。``global_batch_size`` 小于单轮样本数时一轮 rollout 会更新多次，后续 micro-batch 相对已更新策略打分，KL 会明显上升；在确认学习信号之前建议每轮只更新一到两次。

GPU 放置
~~~~~~~~

LIBERO 的多个子进程在同一 EGL 设备上每步渲染会触发 ``read_pixels`` 的原生 abort（父进程看到 ``EOFError``、``exitcode=-6``）。``env.<split>.render_gpu_ids`` 把每个子进程渲染器钉到一张 GPU（``MUJOCO_EGL_DEVICE_ID`` 轮转），这些 GPU 上不能再跑 CUDA 计算。8 卡节点上 actor 占 0-1、rollout 占 3、env worker 占 2 时，渲染卡最多 4 张，即最多 4 到 6 个环境。

启动命令
~~~~~~~~

.. code-block:: bash

   export PYTHONPATH=$PWD EMBODIED_PATH=$PWD/examples/embodiment
   export ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python examples/embodiment/train_embodied_agent.py \
       --config-name libero_spatial_ppo_openwam_long \
       env.train.total_num_envs=4 +env.train.render_gpu_ids=[4,5,6,7] \
       actor.global_batch_size=128 runner.max_epochs=30 runner.max_steps=30 runner.save_interval=10

需要关注的指标
~~~~~~~~~~~~~~

- ``rollout/success_once``、``rollout/return``：每轮的成功率与回报，是唯一直接反映策略是否变好的量。
- ``actor/ratio``、``actor/approx_kl``、``actor/clip_fraction``：未更新权重时 ratio 为 1.0、KL 为 0；``actor/ratio`` 的表格值是未按 loss mask 归一的均值，会接近 ``loss_mask_fraction``。
- ``critic/explained_variance``：当前 value head 只用 8 个池化标量，解释方差通常接近 0，此时优势估计退化为归一化回报，属于已知限制而非训练故障。
- ``rollout/rewards``、``rollout/returns_max``：actor 侧的 masked 均值，两个 actor rank 分别统计后取平均。

导出与评测
----------

FSDP actor 把整个 ``OpenWAMPolicy`` 的 state dict 保存到 ``<log_path>/<experiment>/checkpoints/global_step_N/actor/model_state_dict/full_weights.pt``。把它重建为 OpenWAM 原生 checkpoint 目录：

.. code-block:: bash

   python toolkits/openwam/export_ppo_checkpoint.py \
       --rlinf-checkpoint <log_path>/<experiment>/checkpoints/global_step_20 \
       --source-checkpoint /path/to/new-openwam-libero-sft-10epoch-delta-aligned \
       --output /path/to/openwam-libero-ppo-step20 --link-assets --verify cuda

导出脚本去掉 ``architecture.`` 前缀、丢弃 ``vlm_backbone.*``（OpenWAM 以目录形式保存 VLM）、校验键集合与源 checkpoint 一致、写出 ``checkpoint_step_N.safetensors``，并把 value head 保存为 ``rlinf_value_head.pt``。``--verify`` 用 ``openwam.deploy.load_from_checkpoint_dir`` 重新加载核对。导出目录可以直接作为 ``rollout.model.model_path`` 传给评测配方：

.. code-block:: bash

   python evaluations/eval_embodied_agent.py --config-name libero_spatial_openwam_eval \
       rollout.model.model_path=/path/to/openwam-libero-ppo-step20 \
       env.eval.total_num_envs=5 env.eval.render_gpu_ids=[2,3,4,5,6] env.eval.rollout_epoch=10

``evaluations/libero/`` 下提供 ``libero_{spatial,object,goal,10}_openwam_eval.yaml`` 四个配方，每个 suite 都沿用 OpenWAM 原生客户端的 600 步上限。日志中每条 episode 都有 ``[libero eval] task_id=.., trial_id=.., success=..``，可以按任务拆分成功率。

当前结果
--------

同一 checkpoint、同一组 trial（10 任务 × 5 trial、5 环境、4 步去噪、horizon 10）下的 LIBERO-Spatial 成功率：

.. list-table::
   :header-rows: 1
   :widths: 40 20 20 20

   * - checkpoint
     - task 2/3/4（训练任务）
     - 其余 7 个任务
     - 总计
   * - SFT 基线
     - 4/15
     - 32/35
     - 36/50
   * - PPO step 10（task 2/3/4，4 环境，30 轮）
     - 5/15
     - 33/35
     - 38/50
   * - PPO step 20
     - 5/15
     - 31/35
     - 36/50
   * - PPO step 30
     - 2/15
     - 33/35
     - 35/50

在每任务 5 个 trial 的分辨率下 PPO 没有可辨的提升，也没有让未训练任务退化。每轮只有 8 条轨迹、value head 信息量有限，是已知的样本效率瓶颈。

已知限制
--------

- 视频流在这一版 PPO 中是确定性条件，动作似然是去噪转移的代理量，不是对视频轨迹积分后的精确边际似然。
- 只有 Wan2.2 双系统在 LIBERO 上跑过完整 PPO，其他架构只做过模型级校验；Cosmos-Predict2.5 未测试。
- 完整 PPO 只在 LIBERO 上跑过。RoboTwin 的评测适配已接入（``evaluations/robotwin/robotwin_{click_bell,place_empty_cup}_openwam_eval.yaml``，见 :doc:`../../evaluations/guides/robotwin`），尚未在真实 RoboTwin 环境里跑过验证。
- 同一 GPU 上多个 LIBERO 渲染子进程会崩溃，限制了单机可用的环境数。
