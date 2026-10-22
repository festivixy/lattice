"""FSDP training loop with gradient accumulation, cosine schedule and async checkpoints."""
import math
import time
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy


def cosine_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * progress))


def wrap(model: torch.nn.Module, precision: str) -> FSDP:
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
    policy = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32, buffer_dtype=dtype)
    return FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=policy,
        use_orig_params=True,
        limit_all_gathers=True,
        device_id=torch.cuda.current_device(),
    )


def train(model, optimizer, loader, cfg, checkpointer, logger):
    rank = dist.get_rank()
    total_steps = cfg.epochs * cfg.steps_per_epoch
    accum = cfg.grad_accum
    step = 0
    model.train()
    for epoch in range(cfg.epochs):
        tokens_seen, started = 0, time.perf_counter()
        for micro, (inputs, targets) in enumerate(loader):
            last_micro = (micro + 1) % accum == 0
            sync = nullcontext() if last_micro else model.no_sync()
            with sync:
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
                )
                (loss / accum).backward()
            tokens_seen += inputs.numel()
            if not last_micro:
                continue

            grad_norm = model.clip_grad_norm_(cfg.max_grad_norm)
            lr = cosine_lr(step, cfg.warmup, total_steps, cfg.lr, cfg.lr * 0.01)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if step % cfg.log_every == 0 and rank == 0:
                elapsed = time.perf_counter() - started
                logger.info(
                    "epoch %d | step %d | loss %.4f | lr %.2e | gnorm %.3f | %.0f tok/s",
                    epoch + 1, step, loss.item(), lr, grad_norm, tokens_seen / elapsed,
                )
            if step % cfg.ckpt_every == 0:
                checkpointer.save_async(model, optimizer, step)
            if step % cfg.steps_per_epoch == 0:
                break
    checkpointer.wait()
    return step
