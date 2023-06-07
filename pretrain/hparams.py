from absl import flags
from utils import tool
from utils.tool import bcolors

# additional hyper-parameters
flags.DEFINE_enum('model', 'SGC', ['SGC'],
help='gnn architecture')
flags.DEFINE_integer('num_epochs', 200, 
help='epoch number of pre-training')
flags.DEFINE_integer('seed', 42, 
help='epoch number of pre-training')
flags.DEFINE_enum('dataset', 'Cora', ['Cora', 'CiteSeer', 'PubMed', "chameleon", "squirrel", \
    "Wisconsin", "Cornell", "Texas", 'Actor'],
help='training dataset')
flags.DEFINE_string("data_dir", "./data",
help="path for loading dataset")

# tunable hparams for generalization
flags.DEFINE_float('weight_decay', 8.5e-6,
help='l2 regularization coeffcient')
flags.DEFINE_float('lr', 0.2, 
help='learning rate')

FLAGS = flags.FLAGS

def get_pretrain_hparams(seed, is_dir=False):
    hparams = [
        FLAGS.lr,
        FLAGS.weight_decay,
        seed,
        ]
    hparams = '_'.join(map(str, hparams))
    
    pretrain_dir = f'./result/{FLAGS.dataset}/{FLAGS.model}/pretrain/{hparams}'
    if is_dir:
        tool.check_dir(pretrain_dir)
        print(f'{bcolors.RED}pretrain dir: {pretrain_dir}{bcolors.END}')
        return pretrain_dir
    else:
        return hparams