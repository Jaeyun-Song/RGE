from absl import flags
import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy

from utils import tool
from pretrain.main import *

FLAGS = flags.FLAGS

loss_fn_jitted = jax.jit(loss_fn, static_argnums=1)

def compute_acc_retrain(trainer, dataset, *args, **kwargs):
    """Retrain NN on the given current graph and Evaluate NN on the original graph"""
    
    graph = dataset['graph']
    label = dataset['label']
    train_mask = dataset['train_mask']
    val_mask = dataset['val_mask']
    test_mask = dataset['test_mask']
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    graph_orig = kwargs['graph_orig']
    
    # Train NN on the current graph
    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)
    trainer = init_trainer_jitted(rng_init, graph, label.shape[1])
    
    new_train_mask = deepcopy(train_mask)
    
    for epoch in range(FLAGS.num_epochs):
        rng, rng_ = jax.random.split(rng)
        log, trainer = opt_step_jitted(
            trainer, 
            graph,
            graph,
            label, 
            new_train_mask, 
            val_mask,
            test_mask,
            rng_,
            )
    
    # Validate NN on the original graph
    rng = jax.random.PRNGKey(42)
    _, log = loss_fn_jitted(
        trainer.params,
        FLAGS.weight_decay,
        trainer,
        graph_orig,
        graph_orig,
        label,
        train_mask,
        val_mask,
        test_mask,
        rng,
        )
    val_acc = log['val_acc']
    test_acc = log['test_acc']
                      
    return val_acc, test_acc, trainer.params