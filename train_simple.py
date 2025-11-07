# train_dinov3_style_minimal.py
import os, math, gc
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.distributed._composable.fsdp as cfsdp
from torch.utils.data import Dataset, DataLoader

# ======== Dummy Dataset ==========
class RandomClassificationDataset(Dataset):
    def __init__(self, n=2000, d=1024, ncls=1000, seed=42):
        g = torch.Generator().manual_seed(seed)
        self.x = torch.randn(n, d, generator=g)
        self.y = torch.randint(0, ncls, (n,), generator=g)
    def __len__(self): return len(self.x)
    def __getitem__(self, i): return self.x[i], self.y[i]

# ======== Simple Model ==========
class TinyViT(nn.Module):
    def __init__(self, d=1024, hidden=2048, ncls=1000):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.GELU(),
            nn.Linear(hidden, ncls)
        )
    def forward(self, x): return self.net(x)

# ======== Cosine LR schedule (DINOv3 style) ==========
class CosineScheduler:
    def __init__(self, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0.0):
        self.base_value = base_value
        self.final_value = final_value
        self.total_iters = total_iters
        self.warmup_iters = warmup_iters
        self.start_warmup_value = start_warmup_value
        self.schedule = self._build()
    def _build(self):
        schedule = []
        for it in range(self.total_iters):
            if it < self.warmup_iters:
                val = self.start_warmup_value + (self.base_value - self.start_warmup_value) * it / self.warmup_iters
            else:
                cosine = 0.5 * (1 + math.cos(math.pi * (it - self.warmup_iters) / (self.total_iters - self.warmup_iters)))
                val = self.final_value + (self.base_value - self.final_value) * cosine
            schedule.append(val)
        return schedule
    def __getitem__(self, idx): return self.schedule[min(idx, len(self.schedule)-1)]

# ======== Main training loop ==========
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if world_size > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # Hyperparams similar to DINOv3
    epochs = 2
    batch_size = 64
    iter_per_epoch = 100
    total_iters = epochs * iter_per_epoch

    # Model, loss, optimizer
    model = TinyViT().to(device)
    if world_size > 1:
        model = cfsdp.fully_shard(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.04)

    # Schedulers (same shape as DINOv3’s)
    lr_schedule = CosineScheduler(1e-3, 1e-6, total_iters, warmup_iters=5)
    wd_schedule = CosineScheduler(0.04, 0.4, total_iters)
    
    # Dataset
    ds = RandomClassificationDataset(n=6400)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

    # Training loop
    print(f"Starting training on rank {local_rank} (world_size={world_size})")
    it = 0
    for epoch in range(epochs):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

            # update lr + wd
            lr, wd = lr_schedule[it], wd_schedule[it]
            for g in optimizer.param_groups:
                g["lr"] = lr
                g["weight_decay"] = wd

            # forward / backward
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            # logging (avg loss across ranks)
            if world_size > 1:
                loss_tensor = loss.detach().clone()
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                loss_val = loss_tensor.item()
            else:
                loss_val = loss.item()

            if dist.get_rank() == 0 and it % 10 == 0:
                print(f"[Iter {it:04d}] loss={loss_val:.4f} lr={lr:.6f} wd={wd:.4f}")

            it += 1
            if it >= total_iters:
                break

    if world_size > 1:
        dist.destroy_process_group()
    if local_rank == 0:
        print("Training completed successfully.")

if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()