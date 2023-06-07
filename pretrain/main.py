from datasets import dataloader

import functools
import jax
import jax.numpy as jnp
import optax
import numpy as np
from typing import OrderedDict

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from utils import tool
from models import sgc
from pretrain.hparams import get_pretrain_hparams
from copy import deepcopy
import os

from absl import app, flags
FLAGS = flags.FLAGS

def init_trainer(rng, batch, num_classes):
    model = sgc.SGC
    
    net = functools.partial(model, num_classes=num_classes)
    forward = tool.make_forward(net)
    params, state = forward.init(rng, batch, train=True, print_shape=False)
    tx = optax.adam(FLAGS.lr)
    trainer = tool.Trainer.create(
        apply_fn=forward.apply, 
        params=params,
        state=state,
        tx=tx,
    )
    return trainer

init_trainer_jitted = jax.jit(init_trainer, static_argnums=2)

def loss_fn(params, wd_coef, trainer, graph, train_graphs, label, train_mask, val_mask, test_mask, rng):
    new_params, unravel_fn = tool.params_to_vec(params, True)

    # Compute train accuracy and loss on the current graph
    logit = tool.forward(unravel_fn(new_params), trainer, train_graphs, rng=rng, train=True)
    log_prob = jax.nn.log_softmax(logit)
    
    train_loss = - ((label * log_prob).sum(axis=-1) * train_mask).sum() / train_mask.sum()
    train_acc = ((jnp.argmax(logit,axis=-1)==jnp.argmax(label, axis=-1)) * train_mask).sum()/ train_mask.sum()
    
    # Compute val/test accuracy and loss on the orignal graph
    logit_eval = tool.forward(unravel_fn(new_params), trainer, graph, rng=rng, train=False)
    log_prob_eval = jax.nn.log_softmax(logit_eval)
    
    val_loss = - ((label * log_prob_eval).sum(axis=-1) * val_mask).sum() / val_mask.sum()
    val_acc = ((jnp.argmax(logit_eval,axis=-1)==jnp.argmax(label, axis=-1)) * val_mask).sum()/ val_mask.sum()
    
    test_loss = - ((label * log_prob_eval).sum(axis=-1) * test_mask).sum() / test_mask.sum()
    test_acc = ((jnp.argmax(logit_eval,axis=-1)==jnp.argmax(label, axis=-1)) * test_mask).sum()/ test_mask.sum()

    # Apply weight decay
    wd = 0.5 * (new_params**2).sum()
    loss = train_loss.mean() + wd_coef * wd
    log = [
        ('tr_loss', train_loss),
        ('tr_acc', train_acc),
        ('val_loss', val_loss),
        ('val_acc', val_acc),
        ('test_loss', test_loss),
        ('test_acc', test_acc),
    ]
    log = OrderedDict(log)
    return loss, log

def opt_step(trainer, graph, train_graphs, label, train_mask, val_mask, test_mask, rng):
    vec_params, unravel_fn = tool.params_to_vec(trainer.params, True)

    grad_fn = jax.grad(loss_fn, has_aux=True)
    # compute grad
    grad, log_before = grad_fn(
        trainer.params,
        FLAGS.weight_decay,
        trainer,
        graph,
        train_graphs,
        label,
        train_mask,
        val_mask,
        test_mask,
        rng,
        )
    grad = tool.params_to_vec(grad)
    grad_norm = jnp.sqrt((grad**2).sum())
    
    # update NN
    trainer = trainer.apply_gradients(grads=unravel_fn(grad))
    return log_before, trainer

opt_step_jitted = jax.jit(opt_step)

def train(seed: int, split_num: int):
    pretrain_dir = get_pretrain_hparams(seed, is_dir=True)
    
    # Get a candidate graph and label to initialize the network.
    dataset_name = FLAGS.dataset
    dataset_path = FLAGS.data_dir
    
    torch_dataset = dataloader.load_graph_from_torch(dataset_name, dataset_path)
    n_cls = torch_dataset[0].y.max().item() + 1
    train_mask, val_mask, test_mask = dataloader.get_split_mask(torch_dataset, dataset_name, dataset_path, split_num=split_num)
    graphs, labels = dataloader.get_single_graph_tuples(torch_dataset)
        
    # Transform impure `net_fn` to pure functions with hk.transform.
    rng = jax.random.PRNGKey(seed)
    rng, rng_init = jax.random.split(rng)
    trainer = init_trainer_jitted(rng_init, graphs, n_cls)

    for epoch in range(FLAGS.num_epochs):
        rng, rng_ = jax.random.split(rng)
        log, trainer = opt_step_jitted(
            trainer, 
            graphs,
            graphs,
            labels,
            train_mask, 
            val_mask,
            test_mask,
            rng_,
            )
        train_loss = log['tr_loss']
        train_acc = log['tr_acc']
        val_loss = log['val_loss']
        val_acc = log['val_acc']
        test_loss = log['test_loss']
        test_acc = log['test_acc']
        
        if epoch % 20 == 0 or epoch == (FLAGS.num_epochs - 1):
            print(f'epoch: {epoch}, train_loss: {train_loss:.3f}, '
                    f'train_acc: {train_acc:.3f}, val_loss: {val_loss:.3f}, '
                    f'val_acc: {val_acc:.3f}')
        
    print('Training finished')
    tool.save_ckpt(trainer, pretrain_dir)
    print(f'test_loss: {val_acc:.3f}, test_acc: {test_acc:.3f}')
    return val_acc, test_acc

def main(_):
    seed = 42
    val_acc_list = []
    test_acc_list = []
    if FLAGS.dataset in ["Cornell", "Wisconsin", "Texas"]:
        num_repetition = 100
    else:
        num_repetition = 10
    for i in range(num_repetition):
        val_acc, test_acc = train(seed=seed, split_num=i)
        test_acc_list.append(test_acc)
        val_acc_list.append(val_acc)
        seed += 1
    print(f'val_acc: {np.mean(val_acc_list):.4f} ± {np.std(val_acc_list)/np.sqrt(num_repetition):.4f}')
    print(f'test_acc: {np.mean(test_acc_list):.4f} ± {np.std(test_acc_list)/np.sqrt(num_repetition):.4f}')

if __name__ == "__main__":
    app.run(main)