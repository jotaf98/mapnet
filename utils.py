
from torch.utils.data.sampler import Sampler, BatchSampler
import json


def rotate90(tensor, times):
  """Rotate tensor clockwise by 90 degrees a number of times.
  Assumes spatial dimensions are the last ones."""
  if times == 1:  # 90 deg
    return tensor.transpose(-2, -1).flip(-1)
  elif times == 2:  # 180 deg
    return tensor.flip(-2).flip(-1)
  elif times == 3:  # 270 deg
    return tensor.transpose(-2, -1).flip(-2)
  else:  # 0 deg, no change
    assert times == 0
    return tensor


# also see: https://github.com/ActiveState/code/blob/3b27230f418b714bc9a0f897cb8ea189c3515e99/recipes/Python/578112_Bresenhams_line_algorithm/recipe-578112.py
def bresenham_line(x, y, x2, y2):
  """Brensenham line algorithm"""
  steep = 0
  coords = []
  dx = abs(x2 - x)
  if (x2 - x) > 0: sx = 1
  else: sx = -1
  dy = abs(y2 - y)
  if (y2 - y) > 0: sy = 1
  else: sy = -1
  if dy > dx:
    steep = 1
    x,y = y,x
    dx,dy = dy,dx
    sx,sy = sy,sx
  d = (2 * dy) - dx
  for i in range(0,dx):
    if steep: coords.append((y,x))
    else: coords.append((x,y))
    while d >= 0:
      y = y + sy
      d = d - (2 * dx)
    x = x + sx
    d = d + (2 * dy)
  coords.append((x2,y2))
  return coords


def sub2ind(sub, shape, check_bounds=True):
  """Tensor subscripts to linear index"""
  #assert len(shape) == 2  # 2D test
  #idx = sub[0] * shape[1] + sub[1]
  #return idx

  idx = 0   # if isinstance(sub[0], int) else sub[0].new_zeros(())
  stride = 1
  for (i, n) in reversed(list(zip(sub, shape))):
    if check_bounds:
      if isinstance(i, int): assert i >= 0 and i < n
      else: assert (i >= 0).all() and (i < n).all()
    idx += i * stride
    stride *= n
  return idx


def ind2sub(idx, shape):
  """Linear index to tensor subscripts"""
  #assert len(shape) == 2  # 2D test
  #sub = [idx // shape[1], idx % shape[1]]
  #return sub

  sub = [None] * len(shape)
  for (s, n) in reversed(list(enumerate(shape))):  # write subscripts in reverse order
    sub[s] = idx % n
    idx = idx // n
  return sub



def test_ind2sub():
  """Test ind2sub/sub2ind"""
  import torch as t

  shape = [3, 4, 5]
  n = shape[0] * shape[1] * shape[2]
  #shape = [3, 4]  # 2D test (easier)
  #n = shape[0] * shape[1]

  # test invertibility
  for idx in range(n):
    sub = ind2sub(idx, shape)
    idx_ = sub2ind(sub, shape)
    assert idx_ == idx
  print('ind2sub/sub2ind: Passed invertibility test.')

  # test consistency with pytorch tensor dimension ordering
  A = t.zeros(n)
  for idx in range(n):
    A.zero_()
    A[idx] = 1.0

    B = t.reshape(A, shape)
    sub_ = t.nonzero(B)

    assert sub_.numel() == len(shape)
    sub_ = sub_.tolist()[0]

    sub = ind2sub(idx, shape)

    assert sub_ == sub
  print('ind2sub/sub2ind: Passed PyTorch consistency test.')


def test_rotate90():
  """Test rotate90"""
  import torch as t
  from math import cos, sin, pi

  def rotate_vec(vec=(-0.5, -0.5), center=(0.5, 0.5), times=0):
    """Rotate vector around center coordinates a number of times, in 90 degrees increments, with integer outputs"""
    (vx, vy) = vec
    (cx, cy) = center
    a = times * pi / 2
    x = cx + vx * cos(a) - vy * sin(a)
    y = cy + vx * sin(a) + vy * cos(a)
    return (round(x), round(y))

  # note: rotate_vec rotates clockwise, in *cartesian* coordinates (Y points up).
  # tensor/matrix subscripts are laid out with a flipped vertical axis (Y points down).
  # so, to achieve a clockwise rotation of tensor subscripts, rotate the coordinates
  # (rotate_vec) in the opposite direction (negate the angle/number of rotations).

  A = t.zeros((2, 2))
  A[0,0] = 1.0  # one-hot

  B = rotate90(A, 0)
  assert B[0,0].item() == 1.0
  assert B[rotate_vec(times=0)].item() == 1.0
  
  B = rotate90(A, 1)
  assert B[0,1].item() == 1.0
  assert B[rotate_vec(times=-1)].item() == 1.0

  B = rotate90(A, 2)
  assert B[1,1].item() == 1.0
  assert B[rotate_vec(times=-2)].item() == 1.0

  B = rotate90(A, 3)
  assert B[1,0].item() == 1.0
  assert B[rotate_vec(times=-3)].item() == 1.0

  print('rotate90: Passed tests.')


if __name__ == '__main__':
  # run tests
  test_ind2sub()
  test_rotate90()
