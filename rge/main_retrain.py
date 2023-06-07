import jax
import numpy as np
import os
from copy import deepcopy
import pandas as pd
import jraph
import jax.tree_util as tree
from absl import app, flags

from utils import tool
from datasets import dataloader
from pretrain.main import *
from pretrain.hparams import get_pretrain_hparams
from rge.retrain import compute_acc_retrain
from rge.estimate import compute_edge_influence
from rge.group_finder import group_finder
from tqdm import tqdm

FLAGS = flags.FLAGS

def main_influence(SEED, split_num):
    hparams = get_pretrain_hparams(SEED)
    model = FLAGS.model
    dataset_name = FLAGS.dataset
    dataset_path = FLAGS.data_dir
    
    # Load dataset
    save_folder_path = f'./result/{dataset_name}/{model}/retrain/{hparams}'
    tool.check_dir(save_folder_path)
    
    torch_dataset = dataloader.load_graph_from_torch(dataset_name, dataset_path)
    n_cls = torch_dataset[0].y.max().item() + 1
    
    train_mask, val_mask, test_mask = dataloader.get_split_mask(torch_dataset, dataset_name, dataset_path, split_num=split_num)
    graphs, labels = dataloader.get_single_graph_tuples(torch_dataset)

    # Initialize trainer
    rng = jax.random.PRNGKey(SEED)
    rng, rng_init = jax.random.split(rng)
    trainer = init_trainer_jitted(rng, graphs, n_cls)
    trainer = tool.load_ckpt(f'result/{dataset_name}/{model}/pretrain/{hparams}', trainer)

    dataset = {
        'graph':graphs,
        'label':labels,
        'train_mask':train_mask,
        'val_mask':val_mask,
        'test_mask':test_mask,
    }

    # Compute the degree of a graph    
    degree = np.array(jax.ops.segment_sum(jnp.ones(graphs.receivers.shape[0]), graphs.receivers, graphs.n_node[0]) + 1)
    inf_mask = np.isinf(degree)
    degree[inf_mask.astype(bool)] = 1
    avg_degree = np.mean(degree[np.asarray(train_mask).astype(bool)])
    
    # Edge candidate
    effective_edge_index = dataloader.get_effect_edge_idx(graphs, train_mask)
    half_n_edge = graphs.n_edge[0] // 2
    sampled_idx = np.expand_dims(effective_edge_index,axis=1)
    edge_candidate_idx = np.concatenate((sampled_idx,sampled_idx+half_n_edge),axis=1)
    
    dataset['graph'] = deepcopy(graphs)
    cur_graph = deepcopy(graphs)
    
    # removing edges and retrain models on rectified graphs
    max_num = 2000
    num_iters = 0
    best_val_acc = 0
    val_acc_list, test_acc_list = [], []
    np_train_mask = np.array(train_mask).astype(bool)
    while num_iters < max_num:
        assert (cur_graph.receivers[half_n_edge:]-cur_graph.senders[:half_n_edge]).sum() == 0
        assert (cur_graph.receivers[:half_n_edge]-cur_graph.senders[half_n_edge:]).sum() == 0
        
        # Compute edge influence
        dataset['graph'] = deepcopy(cur_graph)
        output = compute_edge_influence(deepcopy(trainer), dataset, \
            graph_orig=deepcopy(graphs), edge_candidate_idx=edge_candidate_idx)
        edge_influence = np.array(output["influence"])
        
        if (edge_influence < 0).astype(int).sum() == 0:
            break
        
        # Find edges to be removed
        negative_edge_idx = group_finder(cur_graph, edge_candidate_idx, edge_influence, train_mask, avg_degree)
        
        # Remove edges from the current graph
        cur_graph = cur_graph._replace(
            n_edge=np.asarray([cur_graph.n_edge[0] - negative_edge_idx.shape[0]]),
            senders=np.delete(cur_graph.senders, negative_edge_idx), 
            receivers=np.delete(cur_graph.receivers, negative_edge_idx), 
        )
        
        effective_edge_index = dataloader.get_effect_edge_idx(cur_graph, train_mask)
        half_n_edge = cur_graph.n_edge[0] // 2
        sampled_idx = np.expand_dims(effective_edge_index, axis=1)
        edge_candidate_idx = np.concatenate((sampled_idx, sampled_idx + half_n_edge), axis=1)
        
        # Retrain NN on the current graph and Update parameters
        dataset['graph'] = deepcopy(cur_graph)
        val_acc, test_acc, params = compute_acc_retrain(trainer, dataset, graph_orig=deepcopy(graphs))
        trainer = trainer.replace(params=params)
        
        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        num_iters += 1
        
        print(f'{num_iters}: val_acc: {val_acc:.4f}, test_acc: {test_acc:.4f}')
        
    # save as tsv    
    df = pd.DataFrame(np.stack([np.array(cur_graph.senders), np.array(cur_graph.receivers)], axis=1), columns=['senders', 'receivers'])
    df.to_csv(os.path.join(save_folder_path,"last_edges.tsv"), sep='\t', index=False)
    
    return num_iters, val_acc, test_acc, dataset_name

def main(_):
    seed = 42
    n_iter_list = []
    val_acc_list = []
    test_acc_list = []
    if FLAGS.dataset in ["Cornell", "Wisconsin", "Texas"]:
        num_repetition = 100
    else:
        num_repetition = 10
    for i in tqdm(range(num_repetition)):
        n_iter, val_acc, test_acc, dataset_name = main_influence(SEED=seed, split_num=i)
        n_iter_list.append(n_iter)
        test_acc_list.append(test_acc)
        val_acc_list.append(val_acc)
        seed += 1
    print(f'{dataset_name}, num_iter: {np.mean(n_iter_list):.4f} ± {np.std(n_iter_list)/np.sqrt(num_repetition):.4f}')
    print(f'{dataset_name}, val_acc: {np.mean(val_acc_list):.4f} ± {np.std(val_acc_list)/np.sqrt(num_repetition):.4f}')
    print(f'{dataset_name}, test_acc: {np.mean(test_acc_list):.4f} ± {np.std(test_acc_list)/np.sqrt(num_repetition):.4f}')

if __name__ == "__main__":
    app.run(main)