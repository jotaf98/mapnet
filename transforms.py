
import math
import torch as t

# needed for transforms_collate
from torch.utils.data import dataloader
from torch._six import container_abcs


# alternative: subclass Tensor. must return Transformation for all methods. see:
# https://stackoverflow.com/a/33272874, https://stackoverflow.com/a/53135377, https://stackoverflow.com/a/19957897
class Transformation:
  """Generic transformation (abstract base class)"""
  def __init__(self, parameters):
    for param in parameters:
      getattr(self, param)  # check for existence
    self.parameters = parameters
    
  def __add__(self, other):
    raise NotImplementedError()
    
  def __neg__(self):
    raise NotImplementedError()

  def __sub__(self, other):
    """Pose difference (first pose relative to the second pose)"""
    return self + (-other)
  
  def stack(self, dim=0):
    """Return as a single tensor, with parameters stacked along given dimension."""
    parameters =  [getattr(self, name) for name in self.parameters]
    return t.stack(parameters, dim=dim)
  
  def position(self, dim=0):
    """Return position as a single tensor, with parameters stacked along given dimension."""
    if 'z' in self.parameters:
      return t.stack((self.x, self.y, self.z), dim=dim)
    else:
      return t.stack((self.x, self.y), dim=dim)
      
  def pos_distance(self, other, squared=False):
    """Euclidean distance in position"""
    delta = self.position(dim=0) - other.position(dim=0)
    sqr_dist = (delta * delta).sum(dim=0)
    if squared:
      return sqr_dist
    return sqr_dist.sqrt()
  
  def apply(self, func, *args, **kwargs):
    """Applies a function or method to all parameters of the transformation."""
    parameters = {name: func(getattr(self, name), *args, **kwargs) for name in self.parameters}
    return type(self)(**parameters)


class Translation2D(Transformation):
  """Translation in 2D."""
  def __init__(self, x, y=None, dim=None):
    if dim is not None:  # pose parameters are stacked along the given dimension of the first argument
      (x, y) = t.unbind(x, dim=dim)
    self.x = x
    self.y = y
    super().__init__(['x', 'y'])

  def __add__(self, other):
    """Pose composition"""
    return Translation2D(other.x + self.x, other.y + self.y)
  
  def __neg__(self):
    """Pose inverse"""
    return Translation2D(-self.x, -self.y)

  def ang_distance(self, other, squared=False):
    """Rotation distance between poses"""
    return t.zeros(())

  def matrix(self, n=4):
    """Return as a transformation matrix"""
    M = t.eye(n)
    M[0,-1] = self.x.item()
    M[1,-1] = self.y.item()
    return M
    

class Rigid2D(Transformation):
  """Translation and rotation in 2D."""
  def __init__(self, x, y=None, ang=None, dim=None):
    if dim is not None:  # pose parameters are stacked along the given dimension of the first argument
      (x, y, ang) = t.unbind(x, dim=dim)
    self.x = x
    self.y = y
    self.ang = ang
    super().__init__(['x', 'y', 'ang'])

  def __add__(self, other):
    """Pose composition"""
    (sx, sy) = rotate2d(self.x, self.y, other.ang)
    return Rigid2D(other.x + sx, other.y + sy, other.ang + self.ang)

  def __neg__(self):
    """Pose inverse"""
    (ix, iy) = rotate2d(self.x, self.y, -self.ang)
    return Rigid2D(-ix, -iy, -self.ang)

  def ang_distance(self, other, squared=False):
    """Rotation distance between poses"""
    # compute shortest distance between angles, with wrap-around between 0 and 2*pi
    delta = t.remainder(self.ang - other.ang + math.pi, 2 * math.pi) - math.pi
    if squared:
      return delta * delta
    else:
      return delta.abs()
  
  def matrix(self, n=4):
    """Return as a transformation matrix"""
    (x, y, ang) = (self.x.item(), self.y.item(), self.ang.item())
    (c, s) = (math.cos(ang), math.sin(ang))
    if n == 3:
      return t.tensor([[c, -s, x],
                       [s,  c, y],
                       [0,  0, 1]])
    elif n == 4:
      return t.tensor([[c, -s, 0, x],
                       [s,  c, 0, y],
                       [0,  0, 1, 0],
                       [0,  0, 0, 1]])
    else:
      raise ValueError('N must be 3 or 4.')


def rotate2d(x, y, ang):
  if isinstance(x, t.Tensor):
    (c, s) = (t.cos(ang), t.sin(ang))
  else:
    (c, s) = (math.cos(ang), math.sin(ang))
  return (c * x - s * y, s * x + c * y)


def collate_transforms(batch):
  """Extended collate function for DataLoader, to aggregate Transforms across samples into a batch."""
  if isinstance(batch[0], container_abcs.Mapping):
    # recurse on dicts
    return {key: collate_transforms([d[key] for d in batch]) for key in batch[0]}
  
  elif isinstance(batch[0], Transformation):
    # handle transformations
    params = {key: dataloader.default_collate([getattr(d, key) for d in batch]) for key in batch[0].parameters}
    return type(batch[0])(**params)

  else:
    # everything else (cannot contain transformations)
    return dataloader.default_collate(batch)


if __name__ == "__main__":
  # test
  #for trial in range(1):
  #  A = Rigid2D(t.tensor(2.0), t.tensor(1.0), t.tensor(-math.pi / 4))
  #  B = Rigid2D(t.tensor(2.0), t.tensor(4.0), t.tensor(0.0))
  for trial in range(100):
    # compare with matrix representation
    A = Rigid2D(t.randn(1), t.randn(1), t.randn(1))
    B = Rigid2D(t.randn(1), t.randn(1), t.randn(1))
  
    BA1 = t.inverse(A.matrix(3)) @ B.matrix(3)
    print(BA1)

    BA2 = B -  A
    print(BA2.matrix(3))
    print(BA2.stack(dim=0))

    assert((BA1 - BA2.matrix(3)).abs().sum() < 1e-3)

    # test invertibility
    A_  = (A + B) - B
    assert(A.pos_distance(A_) < 1e-3)
    assert(A.ang_distance(A_) < 1e-3)
