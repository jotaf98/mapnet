
import math
import torch as t, torch.nn as nn, torch.nn.functional as F
from einops import rearrange

from utils import rotate90


class MapNet(nn.Module):
  """MapNet returns pose estimate tensors from image sequences"""
  def __init__(self, cnn, embedding_size, map_size, improved_padding, orientations=4, lstm_forget_bias=None, hardmax=False, temperature=1.0, debug_vectorization=False):
    super().__init__()

    assert map_size % 2 == 1, 'Map size should be odd'
    assert orientations in [1, 4]

    self.cnn = cnn

    self.rnn = nn.LSTMCell(embedding_size, embedding_size, bias=True)

    # change default LSTM initialization
    noise = 0.1
    self.rnn.weight_hh.data = -noise + 2 * noise * t.rand_like(self.rnn.weight_hh)
    self.rnn.weight_ih.data = -noise + 2 * noise * t.rand_like(self.rnn.weight_ih)
    self.rnn.bias_hh.data = t.zeros_like(self.rnn.bias_hh)  # redundant with bias_ih
    self.rnn.bias_ih.data = -noise + 2 * noise * t.rand_like(self.rnn.bias_ih)

    if lstm_forget_bias:
      assert self.rnn.bias_hh.shape[0] == 4 * embedding_size
      self.rnn.bias_ih.data[embedding_size : 2*embedding_size] = lstm_forget_bias
      
    self.map_size = map_size
    self.embedding_size = embedding_size
    self.orientations = orientations
    self.hardmax = hardmax
    self.temperature = temperature
    self.improved_padding = improved_padding
    self.debug_vectorization = debug_vectorization  # slow! sanity check vectorized ops


  def forward(self, images):
    """images.shape = (batch, time, channels, obs_height, obs_width)
    output.shape = (batch, time, orientations, map_height, map_width)"""

    (batch_sz, seq_length, channels, obs_h, obs_w) = images.shape
    assert obs_w == obs_h and obs_w % 2 == 1
    map_sz = self.map_size
    state = None

    # merge time dimension into batch dimension, and run net
    images_ = rearrange(images, 'b t e h w -> (b t) e h w')
    obs = self.cnn(images_)
    obs = rearrange(obs, '(b t) e h w -> b t e h w', t=seq_length)
    assert obs.shape[-3] == self.embedding_size
    view_range = (obs.shape[-1] - 1) // 2

    # create stack of rotated observations. shape = (batch, time, orientations, embeding, height, width)
    rotated_obs = t.stack([rotate90(obs, times=i) for i in range(self.orientations)], dim=2)

    # initial pose heatmap, one-hot at center of map and first orientation
    pose = t.zeros((batch_sz, self.orientations, map_sz, map_sz), device=images.device)
    pose[:, 0, (map_sz - 1) // 2, (map_sz - 1) // 2] = 1

    (raw_poses, softmax_poses, maps) = ([], [], [])  # pose tensors and maps through time

    for step in range(seq_length - 1):
      # registration: deconvolve pose estimate with rotated observations to get map-sized embedding
      # output shape = (batch, embeding, height, width)
      registered_obs = self.register(pose, rotated_obs[:,step,:,:,:,:])

      # aggregate registered observations into a map, updating the map state
      (map, state) = self.aggregate_rnn(registered_obs, state)

      # localization: convolve map with rotated observations (from next frame) to obtain pose estimate
      pose = self.localize(map, rotated_obs[:,step+1,:,:,:,:], pad=not self.improved_padding)

      if self.temperature != 1.0:
        pose = pose * self.temperature

      # return poses before the softmax
      raw_poses.append(pose)

      # softmax over spatial/orientation dimensions
      pose = self.softmax_pose(pose)

      # option to apply padding after softmax (so edges are ignored)
      if self.improved_padding: pose = F.pad(pose, [view_range] * 4)
      
      # return poses after the softmax, and the map
      softmax_poses.append(pose)
      maps.append(map)

    # aggregate poses (before softmax) over time and apply padding if needed
    raw_poses = t.stack(raw_poses, dim=1)
    if self.improved_padding:
      raw_poses = F.pad(raw_poses, [view_range] * 4)

    return {'raw_poses': raw_poses, 'softmax_poses': t.stack(softmax_poses, dim=1), 'maps': t.stack(maps, dim=1)}
  

  def register(self, pose, rotated_obs):
    """Registration: deconvolve pose estimate with rotated observations to get map-sized embedding"""
    # output shape = (batch, embeding, height, width)
    (batch_sz, orientations, map_sz, map_sz) = pose.shape
    view_range = (rotated_obs.shape[-1] - 1) // 2

    # non-vectorized version, for comparison
    if self.debug_vectorization:
      reg_obs_ = t.empty((batch_sz, self.embedding_size, map_sz, map_sz), device=rotated_obs.device)
      for b in range(batch_sz):  # input channels: orientations; output channels: embedding.
        reg_obs_[b:b+1,:,:,:] = F.conv_transpose2d(pose[b:b+1,:,:,:], rotated_obs[b,:,:,:,:], padding=view_range)

    # use filter groups to apply a different rotated_obs filter bank to each image in batch. batch size will be 1.
    pose_ = rearrange(pose, 'b o h w -> () (b o) h w')  # push batch dimension to input channels
    rotated_obs_ = rearrange(rotated_obs, 'b o e h w -> (b o) e h w')  # same with filter bank
    reg_obs = F.conv_transpose2d(pose_, rotated_obs_, groups=batch_sz, padding=view_range)
    reg_obs = rearrange(reg_obs, '() (b e) h w -> b e h w', b=batch_sz)  # ungroup batch dimension from output channels

    if self.debug_vectorization:
      assert (reg_obs - reg_obs_).abs().max().item() < 1e-4, 'MapNet registration: vectorized op failed check'

    return reg_obs


  def localize(self, map, rotated_obs, pad):
    """Localization: convolve map with rotated observations to obtain pose estimate"""
    view_range = (rotated_obs.shape[-1] - 1) // 2
    batch_sz = map.shape[0]

    # non-vectorized version, for comparison
    if self.debug_vectorization:
      sz = self.map_size - (0 if pad else 2 * view_range)
      pose_ = t.empty((batch_sz, self.orientations, sz, sz), device=map.device)
      for b in range(batch_sz):
        pose_[b:b+1,:,:,:] = F.conv2d(map[b:b+1,:,:,:], rotated_obs[b,:,:,:,:], padding=(view_range if pad else 0))

    # use filter groups to apply a different rotated_obs filter bank to each image
    map_ = rearrange(map, 'b e h w -> () (b e) h w')
    rotated_obs_ = rearrange(rotated_obs, 'b o e h w -> (b o) e h w')
    pose = F.conv2d(map_, rotated_obs_, groups=batch_sz, padding=(view_range if pad else 0))
    pose = rearrange(pose, '() (b o) h w -> b o h w', b=batch_sz)

    if self.debug_vectorization:
      assert (pose - pose_).abs().max().item() < 1e-4, 'MapNet localization: vectorized op failed check'

    return pose


  def aggregate_rnn(self, registered_obs, state):
    """Aggregate registered observations into map, using an LSTM/RNN"""
    # LSTM/RNN update phase. merge spatial dimensions into batch dim, to treat them independently.
    flat_obs = rearrange(registered_obs, 'b e h w -> (b h w) e')
    if state is None:  # first time, initialize state to 0
      state = self.rnn(flat_obs)
    else:
      state = self.rnn(flat_obs, state)
    
    # get LSTM cell and un-merge spatial dimensions again
    hidden_state = state[0]  # get hidden state (could also use cell value)
    map = rearrange(hidden_state, '(b h w) e -> b e h w', h=self.map_size, w=self.map_size)
    return (map, state)
  

  def softmax_pose(self, pose):
    """Softmax pose probability tensor over spatial/orientation dimensions"""
    flat_pose = rearrange(pose, 'b o h w -> b (o h w)')

    if not self.hardmax:
      flat_pose = F.softmax(flat_pose, dim=1)
    else:
      # hard version (one-hot pose), used mainly for unit tests
      flat_pose = t.eye(flat_pose.shape[1])[flat_pose.argmax(dim=1), :]

    return rearrange(flat_pose, 'b (o h w) -> b o h w', h=pose.shape[-2], w=pose.shape[-1])


  def discretize_pose(self, pose, cell_size=1.0):
    """Discretize a continuous pose on the map grid, with the origin in the center of the map."""
    center = (self.map_size - 1) / 2

    bin_x = (pose.x / cell_size + center).round().long()
    bin_y = (pose.y / cell_size + center).round().long()
    bin_ang = (pose.ang * (self.orientations / 2 / math.pi)).round().long() % self.orientations  # angle wraps around

    invalid = ((bin_x < 0) | (bin_x >= self.map_size) | (bin_y < 0) | (bin_y >= self.map_size) | (bin_ang < 0) | (bin_ang >= self.orientations))

    return (bin_x, bin_y, bin_ang, invalid)
    

  def undiscretize_pose(self, bin_x, bin_y, bin_ang, cell_size=1.0):
    """Inverse operation to discretize_pose"""
    center = (self.map_size - 1) / 2

    x = bin_x.float() * cell_size - center
    y = bin_y.float() * cell_size - center
    ang = (bin_ang.float() * (2 * math.pi / self.orientations)) % (2 * math.pi)

    return (x, y, ang)

