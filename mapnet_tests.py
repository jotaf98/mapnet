
# MapNet tests using dummy data and hardmax (for discrete results)

import argparse, math
import torch as t
from mapnet import MapNet
from mazes import extract_view
from transforms import Rigid2D
from utils import sub2ind, ind2sub
from overboard import tshow

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection

def parse_map(input):
  """Convert map encoded as a string (characters: #*.) into a PyTorch array"""
  input = input.split('\n')
  out = t.zeros((3, len(input), len(input[0])))
  for (y, line) in enumerate(input):
    line = line.strip()
    for (x, ch) in enumerate(line):
      if ch == '#':
        out[0,y,x] = 1
      elif ch == '*':
        out[1,y,x] = 1
      else:
        out[2,y,x] = 0  #0.1
  return out


def show_result(map, obs, out):
  """Show MapNet result as figures"""
  if map is not None: tshow(map[0,:,:] - map[1,:,:], title='gt map')  # difference between one-hot features, result will be in {-1,0,1}
  tshow(obs[0,:,0,:,:] - obs[0,:,1,:,:], title='obs')
  tshow(out['softmax_poses'][0,...], title='softmax_poses')
  tshow(out['maps'][0,:,0,:,:] - out['maps'][0,:,1,:,:], title='maps')


def visualize_poses(poses, obs, map_sz, title):
  """Visualize poses/trajectory, and superimpose observations at those poses"""
  # obs.shape = (batch, time, channels, height, width)
  view_range = (obs.shape[-1] - 1) // 2
  plt.figure(title)

  for step in range(len(poses)):
    plt.subplot(int(math.ceil(len(poses) / 8)), min(8, len(poses)), step + 1)

    pose = poses[step]
    pose = Rigid2D(x=pose[0], y=pose[1], ang=pose[2] * math.pi / 2)
    pose = pose.apply(t.tensor).apply(t.Tensor.float)

    # plot observations (top-down view) as a set of rectangles (one per cell)
    for channel in (0, 1):
      # local coordinates of cells, with origin at center of observation
      local_pos = t.nonzero(obs[0,step,channel,:,:]).float() - view_range

      # transform to global coordinates using pose
      local_pos = Rigid2D(x=local_pos[:,1], y=local_pos[:,0], ang=t.zeros(local_pos.shape[0]))
      points = local_pos + pose

      # plot cells: ground for channel 0, wall for channel 1
      rects = [plt.Rectangle((x, y), 1.0, 1.0)
        for (x, y) in zip(points.x.tolist(), points.y.tolist())]

      plt.gca().add_collection(PatchCollection(rects, facecolor='yb'[channel], edgecolor=None, alpha=0.3))

    # plot pose
    plt.scatter(pose.x+.5, pose.y+.5, s=20, c='r', marker='o', edgecolors=None)
    plt.plot([pose.x+.5, pose.x+.5 + math.cos(pose.ang)], [pose.y+.5, pose.y+.5 + math.sin(pose.ang)], 'r')

    # axes config
    plt.axis('square')
    plt.xlim(0, map_sz)
    plt.ylim(map_sz, 0)  # flip vertical axis

    plt.grid(True)
    plt.gca().set_xticks(range(0, map_sz))
    plt.gca().set_yticks(range(0, map_sz))


def visualization_test(vectorization=False):
  """Show observations only, for manual inspection"""
  mapnet = MapNet(cnn=lambda x: x, embedding_size=3, map_size=5,
    aggregator='avg', hardmax=True, improved_padding=True, debug_vectorization=vectorization)

  # get local observations
  obs1 = """.#.
            .*#
            ..."""

  obs2 = """.*#
            .#.
            .#."""

  obs3 = """#..
            *#.
            ..."""
  
  # shape = (batch=1, time, channels=1, height, width)
  obs = [parse_map(o) for o in (obs1, obs2, obs3)]
  obs = t.stack(obs, dim=0).unsqueeze(dim=0)

  # run mapnet
  out = mapnet(obs, debug_output=True)

  # show results
  show_result(None, obs, out)
  

def full_test(exhaustive=True, flip=False, vectorization=False):
  """Test MapNet with toy observations"""

  '''# map with L-shape, ambiguous correlation result in some edge cases
  map = parse_map("""...
                     *..
                     ##*""")'''

  # unambiguous map with only 2 identifiable tiles (allows triangulation)
  map = parse_map("""...
                     *..
                     ..#""")

  # enlarge map by 0-padding
  pad = 3
  map = t.nn.functional.pad(map, [pad] * 4, value=0)

  if flip:  # rotates the map 180 degrees
    map = map.flip(dims=[1, 2])

  if not exhaustive:
    # hand-crafted sequence of poses (x, y, angle)
    poses = [
      (1+1, 1, 0+1),  # center (or around it)
      (0, 2, 2),  # bottom-left
      (2, 2, 0),  # bottom-right
      (2, 0, 1),  # top-right
    ]
  else:
    # exhaustive test of all valid poses
    poses = [(x, y, ang) for x in range(0, 3) for y in range(0, 3) for ang in range(4)]

    # start around center, to build initial map
    #poses.insert(0, (1, 1, 0))
    poses.insert(0, (2, 1, 1))
  
  if flip:  # replace initial direction so it points the other way
    poses[0] = (poses[0][0], poses[0][1], 2)

  # account for map padding in pose coordinates
  poses = [(x + pad, y + pad, ang) for (x, y, ang) in poses]

  # get local observations
  obs = [extract_view(map, x, y, ang, view_range=2) for (x, y, ang) in poses]
  obs = t.stack(obs, dim=0)

  # batch of size 2, same samples
  obs = t.stack((obs, obs), dim=0)

  # run mapnet
  mapnet = MapNet(cnn=lambda i: i, embedding_size=3, map_size=map.shape[-1],
    aggregator='avg', hardmax=True, improved_padding=True, debug_vectorization=vectorization)

  out = mapnet(obs)

  # show results
  print(t.tensor(poses)[1:,:])  # (x, y, angle)
  print((out['softmax_poses'] > 0.5).nonzero()[:,(4,3,2)])

  show_result(map, obs, out)

  if True:  #not exhaustive:
    visualize_poses(poses, obs, map_sz=map.shape[-1], title="Ground truth observations")

    pred_poses = [out['softmax_poses'][0,step,:,:,:].nonzero()[0,:].flip(dims=(0,)).tolist()
      for step in range(len(poses) - 1)]
    pred_poses.insert(0, [1+pad, 1+pad, 0])  # insert map-agnostic starting pose (centered facing right)
    visualize_poses(pred_poses, obs, map_sz=map.shape[-1], title="Observations registered wrt predicted poses")

  # compare to ground truth
  for (step, (x, y, ang)) in enumerate(poses[1:]):
    # place the ground truth in the same coordinate-frame as the map, which is
    # created considering that the first frame is at the center looking right.
    # also move from/to discretized poses.
    gt_pose = Rigid2D(*mapnet.undiscretize_pose(t.tensor(x), t.tensor(y), t.tensor(ang)))

    initial_gt_pose = Rigid2D(*mapnet.undiscretize_pose(*[t.tensor(x) for x in poses[0]]))

    (x, y, ang, invalid) = mapnet.discretize_pose(gt_pose - initial_gt_pose)

    assert x >= 2 and x <= map.shape[-1] - 2 and y >= 2 and y <= map.shape[-1] - 2 and ang >= 0 and ang < 4, "GT poses going too much outside of bounds"

    # probability of each pose, shape = (orientations, height, width)
    p = out['softmax_poses'][0,step,:,:,:]
    assert p[ang,y,x].item() > 0.5  # peak at correct location
    assert p.sum().item() < 1.5  # no other peak elsewhere
    assert (p >= 0).all().item()  # all positive


def discretize_test():
  """Test pose discretization/undiscretization"""
  mapnet = MapNet(cnn=lambda i: i, embedding_size=3, map_size=7,
    aggregator='avg', hardmax=True, improved_padding=True)
  
  # test data: all positions and angles
  (x, y, ang) = t.meshgrid(t.arange(7, dtype=t.float) - 3,
                           t.arange(7, dtype=t.float) - 3,
                           t.arange(4, dtype=t.float) * math.pi / 2)
  poses = Rigid2D(x, y, ang)
  poses = poses.apply(t.Tensor.flatten)

  # discretize and undiscretize
  (bin_x, bin_y, bin_ang, invalid) = mapnet.discretize_pose(poses)
  (x, y, ang) = mapnet.undiscretize_pose(bin_x, bin_y, bin_ang)

  assert (x - poses.x).abs().max().item() < 1e-4
  assert (y - poses.y).abs().max().item() < 1e-4
  assert (ang - poses.ang).abs().max().item() < 1e-4
  assert invalid.sum().item() < 1e-4

  # test flat indexes
  shape = [mapnet.orientations, mapnet.map_size, mapnet.map_size]
  bin_idx = sub2ind([bin_ang, bin_y, bin_x], shape, check_bounds=True)
  (ang, y, x) = ind2sub(bin_idx, shape)

  (x, y, ang) = mapnet.undiscretize_pose(x, y, ang)

  assert (x - poses.x).abs().max().item() < 1e-4
  assert (y - poses.y).abs().max().item() < 1e-4
  assert (ang - poses.ang).abs().max().item() < 1e-4
  assert invalid.sum().item() < 1e-4


def discretize_center_test():
  """Test pose discretization center (0,0 should correspond to center bin of map)"""
  mapnet = MapNet(cnn=lambda i: i, embedding_size=3, map_size=7,
    aggregator='avg', hardmax=True, improved_padding=True)
  
  center = (mapnet.map_size - 1) // 2
  
  # test data: positions and angles around center, excluding boundaries
  pos_range = t.linspace(-0.5, 0.5, 20)[1:-1]
  ang_range = t.linspace(-math.pi/4, math.pi/4, 20)[1:-1]

  (x, y, ang) = t.meshgrid(pos_range, pos_range, ang_range)

  poses = Rigid2D(x, y, ang).apply(t.Tensor.flatten)

  # discretize those poses, they should all map to the center bin
  (bin_x, bin_y, bin_ang, invalid) = mapnet.discretize_pose(poses)

  assert ((bin_x == center).all() and (bin_y == center).all()
    and (bin_ang == 0).all() and not invalid.any())


  # discretize positions and angles just outside center
  (xo, yo, ango) = t.meshgrid(t.tensor([-0.6, 0.6]),
                              t.tensor([-0.6, 0.6]),
                              t.tensor([-0.26*math.pi, 0.26*math.pi]))

  poses = Rigid2D(xo, yo, ango).apply(t.Tensor.flatten)

  (xo, yo, ango, invalid) = mapnet.discretize_pose(poses)

  assert ((xo != center).all() and (yo != center).all() and
    (ango != 0).all() and not invalid.any())


  # undiscretize center bin
  (xc, yc, angc) = mapnet.undiscretize_pose(t.tensor(center), t.tensor(center), t.tensor(0))
  assert xc == 0 and yc == 0 and angc == 0


if __name__ == '__main__':
  # parse command line args
  parser = argparse.ArgumentParser()
  parser.add_argument("--exhaustive", action="store_true", help="Check all positions and rotations")
  parser.add_argument("--flip", action="store_true", help="Test with 180-degrees-rotated world, also rotating initial pose 180 degrees")
  parser.add_argument("--visualization-only", action="store_true", help="Simpler test with visualization only (no checking)")
  parser.add_argument("--discretize", action="store_true", help="Test discretization bounds")
  args = parser.parse_args()

  # run tests
  if args.visualization_only:
    visualization_test(vectorization=True)
  elif args.discretize:
    discretize_test()
    discretize_center_test()
  else:
    full_test(exhaustive=args.exhaustive, flip=args.flip, vectorization=True)

  print("Done.")
  plt.show()
  input()  # keep tensor figures open until some input is given
