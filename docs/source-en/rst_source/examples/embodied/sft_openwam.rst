OpenWAM Supervised Fine-Tuning
===============================

Use this recipe to fine-tune OpenWAM on the native LIBERO dataset through RLinf's Ray-managed FSDP runner. See :doc:`openwam` for PPO training. The recipe loads OpenWAM's checkpoint directory, reuses its native dataloader and joint video/action loss, and trains the unfrozen modules with full-shard FSDP.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Model
      :text-align: center

      OpenWAM Wan2.2

   .. grid-item-card:: Method
      :text-align: center

      Full-parameter SFT

   .. grid-item-card:: Data
      :text-align: center

      Native LIBERO reader

   .. grid-item-card:: Hardware
      :text-align: center

      2+ GPUs with FSDP

The OpenWAM checkpoint supplies the model and dataloader settings. Set ``data.train_data_paths`` to the dataset root (or a list of roots: each is read with the same dataloader settings and the windows are concatenated, so the mixture is sampled in proportion to size); the loader reads ``config.yaml`` from ``actor.model.model_path`` and keeps the native frame, action, and normalization conventions.

Installation
------------

Install the OpenWAM environment and RLinf:

.. code:: bash

   bash requirements/install.sh embodied --model openwam --env libero
   source .venv/bin/activate

Run It
------

Set the checkpoint and dataset paths in ``examples/sft/config/model/openwam.yaml`` and ``examples/sft/config/libero_sft_openwam.yaml``. The checked-in recipe maps the actor to GPUs ``0-1`` and uses ``use_orig_params: true`` because OpenWAM freezes part of its backbone while training the action modules.

Start the Ray-managed FSDP runner:

.. code:: bash

   bash examples/sft/run_vla_sft.sh libero_sft_openwam

Override ``cluster.component_placement.actor`` and keep ``actor.global_batch_size`` divisible by the actor world size when you change the GPU count.

The preset loads the weights in fp32 (``precision: fp32``) so the optimizer keeps fp32 master weights while FSDP computes in bf16 (``mixed_precision.param_dtype``); bf16 master weights would round away almost every update at ``lr: 1e-6``. The policy is a single FSDP unit (OpenWAM's joint denoising driver reads block weights outside their forward), so the recipe uses FSDP2, which does not keep FSDP1's full-precision unsharded flat parameter. The model preset also keeps ``load_to_device: false``: every rank builds the policy on the CPU and FSDP moves its shard to the GPU while wrapping, so no rank ever holds the whole model on its device. The evaluation recipes load straight onto the GPU with ``load_to_device: true``.

Validation and resuming
-----------------------

Set ``data.val_data_paths`` (one dataset root or a list, read with the same dataloader settings) and ``runner.val_check_interval`` to report ``eval/loss``, ``eval/loss_video`` and ``eval/loss_action`` averaged over the validation loader. ``actor.eval_batch_size`` sets the per-rank validation batch and ``actor.eval_max_batches`` caps the number of validation batches per rank for large datasets. LeRobot-style readers select episodes by split: validation reads the ``val`` split by default, so set ``data.openwam_val_split`` to ``train`` when the validation root is a separate held-out dataset that only ships a train split (an empty validation set is rejected at start-up).

Checkpoints store the data loader, sampler (including the shuffle epoch) and RNG states next to the model weights, so ``runner.resume_dir=<log_path>/<experiment_name>/checkpoints/global_step_<N>`` continues with the next unseen batch instead of restarting the epoch.

Visualization and Results
-------------------------

Monitor ``train/loss``, ``train/loss_video``, and ``train/loss_action`` in TensorBoard. RLinf writes FSDP model and optimizer shards under ``runner.logger.log_path/<experiment_name>/checkpoints/global_step_<N>/actor``.

Evaluation
----------

The LIBERO evaluation recipe uses the same checkpoint contract as this recipe. OpenWAM checkpoints trained on the canonical LIBERO bucket emit absolute EEF10 goals; RLinf converts each goal against the current achieved pose before sending the 7-D OSC command.

Run the short smoke recipe (one environment, 30 steps, i.e. three 10-step generations) after setting ``MUJOCO_GL=egl`` and ``PYOPENGL_PLATFORM=egl``. The recipes place the env worker and rollout worker on separate GPUs so EGL rendering does not share a GPU with OpenWAM inference:

.. code-block:: bash

   python evaluations/eval_embodied_agent.py \
     --config-path ../tests/e2e_tests/evaluations --config-name libero_spatial_openwam_eval

Keep ``env.eval.max_steps_per_rollout_epoch`` divisible by ``rollout.model.num_action_chunks`` (10 with the default ``openwam.inference_horizon``) when you shorten a recipe.

For a checkpoint trained with native delta EEF10 actions, set ``env.eval.openwam_action_representation=native_delta_eef10``. Keep this value aligned with the dataset metadata and checkpoint used for evaluation.


Alternate video encoders
------------------------

Some older OpenWAM checkpoints use legacy encoder names (``vjepa2_1``, ``flux_vae``, or ``wan_vae``); RLinf normalizes these names at deployment. If the encoder weights are stored outside the checkpoint, set ``rollout.model.encoder_model_path`` to the local encoder directory. RLinf stages a temporary config and leaves the checkpoint unchanged.
