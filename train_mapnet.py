
import argparse, os, random
from time import time

import torch as t
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler, BatchSampler

from einops import rearrange
from overboard import Logger

from mapnet import MapNet
from mazes import Mazes
from cnn_models import get_two_layers_cnn
from transforms import Rigid2D
from utils import sub2ind, ind2sub


def batch_forward(inputs, mapnet, phase, device, args, logger):
  """Apply MapNet to input sequences, and compute loss and metrics"""
  start = time()
  log_visualizations = logger.rate_limit(10)  # save debugging information every few seconds

  # get inputs
  images = inputs['images'].to(device, non_blocking=True).float()
  abs_gt_poses = Rigid2D(inputs['poses'].to(device, non_blocking=True), dim=2)

  # make ground truth poses relative to the first frame.
  # gt_poses.x/y/ang.shape = (batch, time - 1)
  gt_poses = abs_gt_poses.apply(lambda x: x[:,1:]) - abs_gt_poses.apply(lambda x: x[:,0:1])

  # convert continuous poses to discrete bins. shape = (batch, time - 1)
  (bin_x, bin_y, bin_ang, invalid) = mapnet.discretize_pose(gt_poses)

  # linearize indexes. range = [0, orientations * height * width - 1]
  gt_bins = sub2ind([bin_ang, bin_y, bin_x], [mapnet.orientations, mapnet.map_size, mapnet.map_size], check_bounds=False)

  # run MapNet on images, obtaining predicted poses over time
  # shape = (batch, time - 1, orientations, height, width)
  results = mapnet(images)

  # loss where the correct class is the ground-truth bin, based on pre-softmax scores
  pose_scores = results['raw_poses']  # poses before softmax
  pose_scores_flat = rearrange(pose_scores, 'b t o h w -> (b t) (o h w)')  # merge batch/time dims into a samples dim
  gt_bins[invalid] = -1  # indexes to ignore
  loss = t.nn.functional.cross_entropy(pose_scores_flat, gt_bins.flatten(), ignore_index=-1, reduction='sum') / pose_scores_flat.shape[0]


  # compute metrics
  with t.no_grad():
    # compute accuracy
    pred_idx = pose_scores_flat.argmax(dim=1)

    accuracy = (pred_idx == gt_bins.flatten()).float().mean()

    # compute position and angle error
    (pred_ang, pred_y, pred_x) = ind2sub(pred_idx, [mapnet.orientations, mapnet.map_size, mapnet.map_size])
    pred_poses = Rigid2D(*mapnet.undiscretize_pose(pred_x, pred_y, pred_ang))
    
    pred_poses = pred_poses.apply(rearrange, '(b t) -> b t', b=images.shape[0])
    pos_error = pred_poses.pos_distance(gt_poses).median(dim=1).values.mean()  # median error over each sequence
    ang_error = pred_poses.ang_distance(gt_poses).median(dim=1).values.mean()

  # debug info
  if log_visualizations:
    n = 8  # limit number of samples
    seq_length = images.shape[1]
    
    (gt_x, gt_y, gt_ang, invalid) = mapnet.discretize_pose(gt_poses)
    gt_x[invalid] = 0
    gt_y[invalid] = 0
    gt_ang[invalid] = 0
    for step in range(seq_length - 1):
      obs = images[:n,step,0,:,:] - images[:n,step,1,:,:]
      logger.tensor('obs' + str(step), obs.cpu())

      pose = results['softmax_poses'][:n,step,:,:,:]
      logger.tensor('pose' + str(step), pose.cpu())

      gt_tensor = t.zeros_like(pose)
      gt_tensor[t.arange(n, dtype=t.long), gt_ang[:n,step], gt_y[:n,step], gt_x[:n,step]] = 1
      logger.tensor('gt' + str(step), gt_tensor.cpu())
      
      logger.tensor('map' + str(step), results['maps'][:n,step,:,:,:].cpu())
    
    obs = images[:n,seq_length-1,0,:,:] - images[:n,seq_length-1,1,:,:]
    logger.tensor('obs' + str(seq_length-1), obs.cpu())

    logger.tensor('mapdiff', (results['maps'][:n,-1,:,:,:] - results['maps'][:n,0,:,:,:]).abs().cpu())


  # save metrics
  vars = locals()
  stats = {phase + '.' + name: vars[name].item() for name in ['loss', 'accuracy', 'pos_error', 'ang_error']}

  if logger.avg_count.get(phase + '.loss', 0) >= 2:  # skip first 2 iterations (warm-up time)
    stats[phase + '.time'] = time() - start
    stats[phase + '.sampling_time'] = inputs['time'].sum().item() / max(1, args.data_loaders)

  logger.update_average(stats)
  return loss


def main():
  # parse command line options
  parser = argparse.ArgumentParser()  

  parser.add_argument("experiment", nargs='?', default="", help="Experiment name (sub-folder for this particular run). Default: test")
  parser.add_argument("-data-dir", default='data/maze/', help="Directory where maze data is located")
  parser.add_argument("-output-dir", default='data/mapnet', help="Output directory where results will be stored (point OverBoard to this location)")
  parser.add_argument("-device", default="cuda:0", help="Device, cpu or cuda")
  parser.add_argument("-data-loaders", default=8, type=int, help="Number of asynchronous worker threads for data loading")
  parser.add_argument("-epochs", default=40, type=int, help="Number of training epochs")
  parser.add_argument("-bs", default=100, type=int, help="Batch size")
  parser.add_argument("-lr", default=1e-3, type=float, help="Learning rate")
  parser.add_argument("--no-bn", dest="bn", action="store_false", help="Disable batch normalization")
  parser.add_argument("-seq-length", default=5, type=int, help="Sequence length for unrolled RNN (longer creates more long-term maps)")
  parser.add_argument("-map-size", default=15, type=int, help="Spatial size of map memory (always square)")
  parser.add_argument("-embedding", default=16, type=int, help="Size of map embedding (vector stored in each map cell)")
  parser.add_argument("--no-improved-padding", dest="improved_padding", action="store_false", help="Disable improved padding, which ensures softmax is only over valid locations and not edges")
  parser.add_argument("-lstm-forget-bias", default=1.0, type=float, help="Initial value for LSTM forget gate")
  parser.add_argument("-max-speed", default=0, type=int, help="If non-zero, only samples trajectories with this maximum spatial distance between steps")
  parser.add_argument("--spawn", action="store_true", help="Use spawn multiprocessing method, to work around problem with some debuggers (e.g. VSCode)")
  
  parser.set_defaults(bn=True, improved_padding=True)
  args = parser.parse_args()

  if not t.cuda.is_available(): args.device = 'cpu'

  if args.spawn:  # workaround for vscode debugging
    import torch.multiprocessing as multiprocessing
    multiprocessing.set_start_method('spawn', True)

  if not args.experiment: args.experiment = 'test'
  
  # complete directory with experiment name
  args.output_dir = (args.output_dir + '/' + args.experiment)

  if os.path.isdir(args.output_dir):
    input('Directory already exists. Press Enter to overwrite or Ctrl+C to cancel.')

  # repeatable random sequences hopefully
  random.seed(0)
  t.manual_seed(0)

  # initialize dataset
  env_size = (21, 21)
  full_set = Mazes(args.data_dir + '/mazes-10-10-100000.txt', env_size,
    seq_length=args.seq_length, max_speed=args.max_speed)

  (train_set, val_set) = t.utils.data.random_split(full_set, (len(full_set) - 5000, 5000))

  val_loader = DataLoader(val_set, batch_size=10 * args.bs, shuffle=False, num_workers=args.data_loaders)


  # create base CNN and MapNet
  cnn = get_two_layers_cnn(args)
  mapnet = MapNet(cnn=cnn, embedding_size=args.embedding, map_size=args.map_size,
    lstm_forget_bias=args.lstm_forget_bias, improved_padding=args.improved_padding, orientations=4)

  # use GPU if needed
  device = t.device(args.device)
  mapnet.to(device)

  # create optimizer
  optimizer = t.optim.Adam(mapnet.parameters(), lr=args.lr)

  with Logger(args.output_dir, meta=args) as logger:
    for epoch in range(args.epochs):
      # refresh subset of mazes every epoch
      train_sampler = BatchSampler(RandomSampler(SequentialSampler(range(95000)), num_samples=10000, replacement=True), batch_size=args.bs, drop_last=True)
      train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.data_loaders)

      # training phase
      mapnet.train()
      for inputs in train_loader:
        #with t.autograd.detect_anomaly():

        optimizer.zero_grad()
        loss = batch_forward(inputs, mapnet, 'train', device, args, logger)

        loss.backward()
        optimizer.step()

        logger.print(prefix='train', line_prefix=f"ep {epoch+1} ")

      # validation phase
      mapnet.eval()
      with t.no_grad():
        for inputs in val_loader:
          loss = batch_forward(inputs, mapnet, 'val', device, args, logger)
          logger.print(prefix='val', line_prefix=f"ep {epoch+1} ")

      logger.append()

      # save state
      state = {'epoch': epoch, 'state_dict': mapnet.state_dict(), 'optimizer': optimizer.state_dict()}
      try: os.replace(args.output_dir + "/state.pt", args.output_dir + "/prev_state.pt");
      except: pass
      t.save(state, args.output_dir + "/state.pt")


if __name__ == "__main__":
  main()
