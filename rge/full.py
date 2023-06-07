import jax
import jax.numpy as jnp
import numpy as np
from copy import deepcopy

from utils import tool

from absl import flags
FLAGS = flags.FLAGS

def loss_fn(params, wd_coef, trainer, graph, label, mask, rng):
    logit = tool.forward(params, trainer, graph, rng=rng, train=True)
    log_prob = jax.nn.log_softmax(logit)
    loss = - ((label * log_prob).sum(axis=-1) * mask).sum() / mask.sum()
    wd = wd_coef * (tool.params_to_vec(params)**2).sum()
    return loss + wd


def compute_edge_influence_full(trainer, dataset, *args, **kwargs):
    """Compute current graph influence on validation set with a edge removal (Left term in the equation)"""
    
    graph = dataset['graph']
    label = dataset['label']
    train_mask = dataset['train_mask']
    val_mask = dataset['val_mask']
    
    rng = jax.random.PRNGKey(42)
    rng, rng_ = jax.random.split(rng)
    
    edge_candidate_idx = kwargs['edge_candidate_idx']
    graph_orig = kwargs['graph_orig']
    
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)
    batch = {'x': graph_orig, 'y': label, 'mask': train_mask}
    batch['rng'] = rng_

    def masked_loss_fn(params, batch, mask):
        params = unravel_fn(params)
        logit = tool.forward(params, trainer, batch['x'], batch['rng'], train=False)
        loss = - (jax.nn.log_softmax(logit) * batch['y'])
        return (loss.sum(axis=-1) * mask).sum()
    
    # compute Hessian
    hess_fn = jax.jit(jax.hessian(loss_fn))
    hess = hess_fn(trainer.params, FLAGS.weight_decay, trainer, graph, label, train_mask, rng_)
    
    hess = np.asarray(hess['sgc/linear']['w']['sgc/linear']['w']) # In our SGC, there is a single weight (['sgc/linear']['w'])
    param_length = vec_params.shape[0]
    hess = np.reshape(hess, (param_length,param_length))
    hess = jnp.asarray(hess)
    
    # compute target dataset gradient on the original graph
    masked_grad_fn_j = jax.jit(jax.grad(masked_loss_fn))
    mv_j = jax.jit(mv_full)

    grad_tgt = masked_grad_fn_j(vec_params, batch, val_mask) 
    grad_tgt = grad_tgt / val_mask.sum()
        
    # invert Hessian
    h_inv = jnp.linalg.inv(hess)
    grad_pre = mv_j(h_inv, grad_tgt)
    
    # compute train set gradient on the current graph with a edge removal
    influence = np.zeros(len(edge_candidate_idx))
    
    # calculate the right term (node influence in the current graph)
    batch['x'] = graph
    grad_rhs = masked_grad_fn_j(vec_params, batch, train_mask) 
    grad_rhs = grad_rhs / train_mask.sum()
    right_term_influence = (grad_pre * grad_rhs).sum()
    
    # Remove a pair of edges from the current graph and compute the left term
    numpy_train_mask = np.array(train_mask).astype(bool)
    for i in range(len(edge_candidate_idx)):
        new_graph = deepcopy(graph)
        new_graph = new_graph._replace(
            n_edge=np.asarray([graph.n_edge[0] - edge_candidate_idx[i].shape[0]]),
            senders=np.delete(new_graph.senders, edge_candidate_idx[i], axis=0), 
            receivers=np.delete(new_graph.receivers, edge_candidate_idx[i], axis=0), 
        )
        batch['x'] = new_graph
        
        grad_lhs = masked_grad_fn_j(vec_params, batch, train_mask) 
        grad_lhs = grad_lhs / train_mask.sum()
        influence[i] = -(grad_pre * grad_lhs).sum() + right_term_influence
        
    output = {}
    output["influence"] = influence
        
    return output


def mv_full(hess, vector):
    result = jnp.matmul(hess, vector)    
    return result