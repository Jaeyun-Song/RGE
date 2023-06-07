from typing import Any, Callable
import os

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

import haiku as hk
import flax
from flax import struct
from flax.training import checkpoints
from flax.training.checkpoints import restore_checkpoint as load_ckpt
import optax

class Trainer(struct.PyTreeNode):
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    tx: Callable = struct.field(pytree_node=False)
    params: Any = None
    state: Any = None
    opt_state: Any = None
    
    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        opt_state = tx.init(params)
        return cls(step=0, apply_fn=apply_fn, params=params, tx=tx, opt_state=opt_state, **kwargs)
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(step=self.step+1, params=new_params, opt_state=new_opt_state, **kwargs)

def make_forward(model):
    
    def _forward(*args, **kwargs):
        return model()(*args, **kwargs)
    
    return hk.transform_with_state(_forward)

def params_to_vec(param, unravel=False):
    vec_param, unravel_fn = ravel_pytree(param)
    if unravel:
        return vec_param, unravel_fn
    else:
        return vec_param
    
def forward(params, trainer, input_, rng=None, train=True):
    res, _ = trainer.apply_fn(params, trainer.state, rng, input_, train)
    return res

def save_ckpt(trainer, path):
    checkpoints.save_checkpoint(path, trainer, trainer.step, overwrite=True)
    
def check_dir(folder_path):
    # save path
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
class bcolors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    VIOLET = '\033[95m'
    CYAN = '\033[96m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
def set_seed(seed, deterministic=False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)
    if deterministic:
        os.environ['TF_DETERMINISTIC_OPS'] = '1' # slow but reproducible