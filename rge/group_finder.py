import numpy as np
    
def group_finder(cur_graph, edge_candidate_idx, edge_influence, train_mask, avg_degree):
    
    # Determine edges to be removed
    negative_edge_idx = np.argsort(edge_influence)
    n = (edge_influence < 0).astype(int).sum()
    negative_edge_idx = negative_edge_idx[:n]
    
    train_idx = np.arange(train_mask.shape[0])[np.array(train_mask).astype(bool)]
    sender_array = np.array(cur_graph.senders)
    receiver_array = np.array(cur_graph.receivers)
    affected_nodes_list = []
    
    num_history = (train_mask.shape[0]//2)
    
    # Get effective node mask for each edge
    for i in range(n):
        # 1-hop nodes from a given edge
        new_train_mask = np.zeros(train_mask.shape[0]).astype(bool)
        new_train_mask[[sender_array[edge_candidate_idx[negative_edge_idx[i]]]]] = True
        new_train_mask[[receiver_array[edge_candidate_idx[negative_edge_idx[i]]]]] = True
        
        # 2-hop nodes from a given edge
        edge_mask = new_train_mask[sender_array].astype(bool) | new_train_mask[receiver_array].astype(bool)
        new_train_mask[[sender_array[edge_mask]]] = True
        new_train_mask[[receiver_array[edge_mask]]] = True
        affected_nodes_list.append(new_train_mask[train_idx])
    affected_nodes = np.stack(affected_nodes_list,axis=0).astype(int)     
    previous_affected_nodes = [np.zeros(affected_nodes.shape[1]).astype(int)] * num_history
    
    # patience mechanism
    patience = np.maximum(np.round(np.log2(avg_degree-1))+3,1).astype(int)
    waiter = 0
    selected_idx = []
    for i in range(n):
        if ((sum(previous_affected_nodes) * affected_nodes[i]).sum() == 0) or waiter > patience:
            previous_affected_nodes.append(affected_nodes[i])
            previous_affected_nodes = previous_affected_nodes[-num_history:]
            selected_idx.append(i)
            waiter = 0
        else:
            waiter += 1

    negative_edge_idx = negative_edge_idx[selected_idx]
    negative_edge_idx = np.reshape(edge_candidate_idx[negative_edge_idx],(-1))
        
    return negative_edge_idx