import nki
import nki.language as nl

@nki.jit
def tensor_maxpool_kernel_(in_tensor, sz_pool):
  """NKI kernel to compute a 2D max-pool operation

  Args:
      in_tensor: an input tensor, of dimensions C x H x W
      sz_pool: integer P representing a (square) pool-window size
  Returns:
      out_tensor: the resulting output tensor, of dimensions C x (H/P) x (W/P)
  """

  # Get input/output dimensions
  sz_p, sz_hin, sz_win = in_tensor.shape
  sz_hout, sz_wout = sz_hin // sz_pool, sz_win // sz_pool
  out_tensor = nl.ndarray((sz_p, sz_hout, sz_wout), dtype=in_tensor.dtype,
                          buffer=nl.shared_hbm)

  # Load input data from external memory to on-chip memory
  in_tile = nl.load(in_tensor)

  # Perform the pooling operation using an access pattern to create a 5D view:
  # [sz_p, sz_hout, sz_wout, sz_pool, sz_pool]
  # The pool dimensions are placed last so we can reduce over them.
  pool_view = in_tile.ap([
    [sz_hin * sz_win, sz_p],      # partition stride
    [sz_pool * sz_win, sz_hout],   # outer row stride (hop by pool rows)
    [sz_pool, sz_wout],            # outer col stride (hop by pool cols)
    [sz_win, sz_pool],             # inner row stride (within pool window)
    [1, sz_pool],                  # inner col stride (within pool window)
  ])
  out_tile = nl.max(pool_view, axis=[3, 4])

  # Store the results back to external memory
  nl.store(out_tensor, value=out_tile)

  return out_tensor


if __name__ == "__main__":
    import torch
    import torch_xla

    device = torch_xla.device()

    # Now let's run the kernel
    POOL_SIZE = 2
    C, HIN, WIN = 2, 6, 6
    HOUT, WOUT = HIN//POOL_SIZE, WIN//POOL_SIZE

    in_tensor = torch.arange(C * HIN * WIN, dtype=torch.bfloat16).reshape(C, HIN, WIN).to(device=device)
    out_tensor = tensor_maxpool_kernel_(in_tensor, POOL_SIZE)

    print(in_tensor, out_tensor) # an implicit XLA barrier/mark-step
