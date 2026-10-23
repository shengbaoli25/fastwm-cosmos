"""Stock update loop with Cosmos16 feature geometry, keeping full30k schedule."""
import time

def run_two_updates(trainer, *, synchronize, clock=time.monotonic):
    """Bound only this loop; retain the stock trainer's full scheduler horizon."""
    import torch

    horizon=trainer.max_steps
    if horizon not in (1000,5000,20000,30000) or trainer.global_step != 0:
        raise ValueError("Probe requires a fresh trainer with the complete configured schedule")
    accelerator = trainer.accelerator
    train_model = (trainer.model if hasattr(trainer.model, 'training_loss')
                   else accelerator.unwrap_model(trainer.model))
    batch=trainer.batch_size;action_dim=getattr(train_model,'action_dim',7)
    iterator = iter(trainer.train_loader)
    updates, losses = [], []
    for _ in range(2):
        synchronize()
        started = clock()
        sample = next(iterator)
        if ("video" in sample or "image" in sample or tuple(sample["video_latents"].shape) != (batch, 16, 3, 28, 56)
                or tuple(sample["first_frame_latents"].shape) != (batch, 16, 1, 28, 56)
                or tuple(sample["action"].shape) != (batch, 32, action_dim)
                or tuple(sample["proprio"].shape) != (batch, 32, action_dim+1)):
            raise ValueError("Each rank must consume the configured feature-only batch and action space")
        trainer.batch_in_epoch += 1
        with accelerator.accumulate(trainer.model):
            train_model = (trainer.model if hasattr(trainer.model, "training_loss")
                           else accelerator.unwrap_model(trainer.model))
            with accelerator.autocast():
                loss, components = train_model.training_loss(sample)
            if not bool(torch.isfinite(loss.detach()).all()):
                raise FloatingPointError("Nonfinite Cosmos cached training loss")
            accelerator.backward(loss)
            if not accelerator.sync_gradients:
                raise RuntimeError("The exact accumulation1 recipe must synchronize every microbatch")
            grad_norm = accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
            trainer.optimizer.step()
            skipped = bool(accelerator.optimizer_step_was_skipped)
            if not skipped:
                trainer.scheduler.step()
            trainer.optimizer.zero_grad(set_to_none=True)
            trainer.global_step += 1
            # Preserve the stock loop's cross-rank metric synchronization.
            global_loss = float(accelerator.gather(loss.detach().float().reshape(1)).mean().item())
            global_components = {key: float(accelerator.gather(
                torch.tensor(float(value), device=loss.device).reshape(1)).mean().item())
                for key, value in components.items()}
            norm = float(accelerator.gather(torch.as_tensor(grad_norm, device=loss.device).reshape(1)).mean().item())
        synchronize()
        losses.append({"loss": float(loss.detach().item()), **components})
        updates.append({"update": trainer.global_step, "seconds": clock() - started,
                        "batch_size": batch, "microbatches": 1, "global_loss": global_loss,
                        "global_components": global_components, "gradient_norm": norm,
                        "optimizer_step_was_skipped": skipped,
                        "lr": [float(group["lr"]) for group in trainer.optimizer.param_groups],
                        "sample_indices": sample["sample_index"].detach().cpu().tolist()})
    if trainer.max_steps != horizon:
        raise RuntimeError("The preflight shortened the formal scheduler horizon")
    return losses, updates
