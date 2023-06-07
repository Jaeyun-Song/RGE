from absl import flags
from rge.full import compute_edge_influence_full

flags.DEFINE_enum('if_method', 'full', ['full'], help='method to estimate influence function')
FLAGS = flags.FLAGS

def compute_edge_influence(trainer, dataset, *args, **kwargs):
    estimate_method = globals()[f'compute_edge_influence_{FLAGS.if_method}']
    return estimate_method(trainer, dataset, *args, **kwargs)