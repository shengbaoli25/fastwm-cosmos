"""Exact cached-data continuation without changing the frozen trainer.

Skip *sharded batches*, not samples: Accelerate must retain its original
end-of-epoch padding. Cached datasets perform no stochastic augmentation.
"""
import random
import numpy as np
import torch


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(snapshot):
    py, npy, cpu, cuda = snapshot
    random.setstate(py)
    np.random.set_state(npy)
    torch.set_rng_state(cpu)
    if cuda:
        torch.cuda.set_rng_state_all(cuda)


def verify_rng_and_sampler(trainer, state, progress):
    rank = trainer.accelerator.process_index
    saved = torch.load(state / f"random_states_{rank}.pkl", map_location="cpu", weights_only=False)
    py, npy, cpu, cuda = rng_snapshot()
    n = saved["numpy_random_seed"]
    if (saved["random_state"] != py or npy[0] != n[0]
            or not np.array_equal(npy[1], n[1]) or npy[2:] != n[2:]
            or not torch.equal(cpu, saved["torch_manual_seed"])
            or len(cuda) != len(saved.get("torch_cuda_manual_seed", []))
            or any(not torch.equal(a, b.cpu()) for a, b in zip(cuda, saved.get("torch_cuda_manual_seed", [])))
            or trainer.accelerator.step != saved["step"]):
        raise ValueError("Native Python/NumPy/Torch/CUDA RNG restore is incomplete")
    sampler = trainer.train_sampler
    if (sampler.epoch_offset != progress["epoch"]
            or sampler.resume_batch_offset != progress["batch_in_epoch"]
            or sampler.seed != 42 or sampler.batch_size != 16 or sampler.num_processes != 4
            or not 0 <= progress["batch_in_epoch"] <= len(trainer.train_loader)):
        raise ValueError("Saved dataloader progress differs from the actual sampler")
    model = trainer.accelerator.unwrap_model(trainer.model)
    module = torch.load(state / "pytorch_model/mp_rank_00_model_states.pt",
                        map_location="cpu", weights_only=False, mmap=True)["module"]
    actual = model.state_dict()
    keys = [k for k in actual if k.endswith("rng_state") or k.endswith("manifest_bytes")]
    if not keys or not any(k.endswith("rng_state") for k in keys):
        raise ValueError("Persistent Force memory RNG missing")
    for k in keys:
        if k not in module or not torch.equal(actual[k].detach().cpu(), module[k]):
            raise ValueError("Force memory RNG/protocol differs from checkpoint: " + k)
    return {"python_numpy_torch_cuda": True, "memory_buffers": keys,
            "epoch": progress["epoch"], "batch_in_epoch": progress["batch_in_epoch"]}


class FirstEpochContinuation:
    def __init__(self, loader, sampler, epoch, batches):
        if not hasattr(loader, "iteration") or not 0 <= batches <= len(loader):
            raise ValueError("Expected an Accelerate sharded loader and valid offset")
        self.loader, self.sampler = loader, sampler
        self.epoch, self.batches, self.first = epoch, batches, True
        sampler.set_epoch_offset(0)
        sampler.clear_resume_batch_offset()

    def __getattr__(self, name):
        return getattr(self.loader, name)

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        if not self.first:
            yield from self.loader
            return
        from accelerate import skip_first_batches
        self.first = False
        snapshot = rng_snapshot()
        shorter = skip_first_batches(self.loader, self.batches)
        shorter.set_epoch(self.epoch)
        # SkipBatchSampler wraps BatchSamplerShard; explicitly propagate to the
        # original sampler instead of depending on an Accelerate version's depth.
        self.sampler.set_epoch(self.epoch)
        iterator = iter(shorter)
        try:
            first = next(iterator)
        except StopIteration:
            restore_rng(snapshot)
            self.loader.set_epoch(self.epoch + 1)
            return
        # Recreating an iterator draws a new worker seed. Do not advance the
        # restored model RNG merely because training moved to a new process.
        restore_rng(snapshot)
        yield first
        yield from iterator
        self.loader.set_epoch(self.epoch + 1)


def install_continuation(trainer):
    trainer.train_loader = FirstEpochContinuation(
        trainer.train_loader, trainer.train_sampler, trainer.epoch, trainer.batch_in_epoch)
