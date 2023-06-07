import jraph
import haiku as hk
from dataclasses import dataclass

w_init = hk.initializers.VarianceScaling(1.0, "fan_avg", "truncated_normal")

@dataclass
class SGC(hk.Module):
    num_classes : int
    
    def __call__(self, graph, train=True, print_shape=False):
        
        if print_shape:
            print(graph.nodes.shape, 'input_shape')

        graph = jraph.GraphConvolution(
            update_node_fn=lambda n: n,
            add_self_edges=True,
        )(graph)
        
        if print_shape:
            print(graph.nodes.shape, 'feat_shape')
        
        graph = jraph.GraphConvolution(
            update_node_fn=lambda n: n,
            add_self_edges=True,
        )(graph)
        
        graph = graph._replace(
            nodes=hk.Linear(self.num_classes, w_init=w_init, with_bias=False)(graph.nodes))
        
        if print_shape:
            print(graph.nodes.shape, 'out_shape')
        
        return graph.nodes