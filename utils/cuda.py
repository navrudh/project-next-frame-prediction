import gc

import torch

BYTES_IN_GB = 1024 ** 3


def memuse():
    return "ALLOCATED: {:>6.3f} ({:>6.3f})  CACHED: {:>6.3f} ({:>6.3f})".format(
        torch.cuda.memory_allocated() / BYTES_IN_GB,
        torch.cuda.max_memory_allocated() / BYTES_IN_GB,
        torch.cuda.memory_reserved() / BYTES_IN_GB,
        torch.cuda.max_memory_reserved() / BYTES_IN_GB,
    )


def cached_variables():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (
                hasattr(obj, "data") and torch.is_tensor(obj.data)
            ):
                yield type(obj), obj.size()
        except:
            pass


def refresh_cuda_memory():
    """
    Re-allocate all cuda memory to help alleviate fragmentation
    """
    # Run a full garbage collect first so any dangling tensors are released
    gc.collect()

    # Then move all tensors to the CPU
    locations = {}
    for obj in gc.get_objects():
        if not isinstance(obj, torch.Tensor):
            continue

        locations[obj] = obj.device
        obj.data = obj.data.cpu()
        if isinstance(obj, torch.nn.Parameter) and obj.grad is not None:
            obj.grad.data = obj.grad.cpu()

    # Now empty the cache to flush the allocator
    torch.cuda.empty_cache()

    # Finally move the tensors back to their associated GPUs
    for tensor, device in locations.items():
        tensor.data = tensor.to(device)
        if isinstance(tensor, torch.nn.Parameter) and tensor.grad is not None:
            tensor.grad.data = tensor.grad.to(device)
