import pickle as pk
import torch
from torch_geometric_signed_directed.data import load_directed_real_data
from sklearn import preprocessing
import utils

dataset_name = 'bitcoin_otc+' # ['telegram', 'bitcoin_alpha', 'bitcoin_otc', 'bitcoin_alpha+', 'bitcoin_otc+'] + synth_datasets
task = 'three_class_digraph' # ['existence', 'three_class_digraph', 'weight_prediction']
normalize = True
synth_datasets = ['DBSM_0.05', 'DBSM_0.08', 'DBSM_0.1',\
                  'Di150_0.2', 'Di150_0.5', 'Di150_0.7',\
                  'Di500_0.2', 'Di500_0.5', 'Di500_0.7']

synth_filenames = ['dataset_nodes500_alpha0.05_beta0.2', 'dataset_nodes500_alpha0.08_beta0.2', 'dataset_nodes500_alpha0.1_beta0.2',\
                   'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',\
                   'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',\
                   'dataset_nodes150_alpha0.6_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue',\
                   'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.2_opposite-signFalse_negative-edgesFalse_directedTrue',\
                   'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.5_opposite-signFalse_negative-edgesFalse_directedTrue',\
                   'dataset_nodes500_alpha0.1_beta0.2_undirected-percentage0.7_opposite-signFalse_negative-edgesFalse_directedTrue']

if dataset_name in ['telegram']:
    data = load_directed_real_data(dataset=dataset_name, name=dataset_name)
    if normalize == True:
        data.edge_weight = torch.exp(-1/data.edge_weight)
elif dataset_name in ['bitcoin_alpha', 'bitcoin_otc', 'bitcoin_alpha+', 'bitcoin_otc+']:
    if dataset_name == 'bitcoin_alpha' or dataset_name == 'bitcoin_otc':
        data = utils.load_signed_real_data_no_negative(dataset=dataset_name, root='./tmp/', keep_negatives=True)
    else:
        data = utils.load_signed_real_data_no_negative(dataset=dataset_name[:-1], root='./tmp/', keep_negatives=False)
    if normalize == True:
        data.edge_weight = torch.tensor(preprocessing.MaxAbsScaler().fit_transform(data.edge_weight.reshape(-1, 1))).T.squeeze()
else:
    synth_filename = synth_filenames[synth_datasets.index(dataset_name)]
    try:
        data = pk.load(open(f'./tmp/synthetic/{synth_filename}.pk','rb'))
        if normalize == True:
            data.edge_weight = torch.tensor(preprocessing.MaxAbsScaler().fit_transform((data.edge_weight.reshape(-1, 1)))).T.squeeze()
    except:
        raise Exception("Wrong dataset.")
        
edge_index = data.edge_index
size = torch.max(edge_index).item() + 1
data.num_nodes = size
datasets = utils.link_class_split_new(data, prob_val=0.05, prob_test=0.15, splits=10, task=task, \
                                      maintain_connect=True, keep_negatives=dataset_name != 'bitcoin_alpha+' and dataset_name != 'bitcoin_otc+')

train = []
train_pos = []
train_neg = []
train_zero = []
val = []
val_pos = []
val_neg = []
val_zero = []
test = []
test_pos = []
test_neg = []
test_zero = []
for i in range(10):
    train.append(len(datasets[i]['train']['label'].squeeze()))
    train_pos.append(torch.count_nonzero(datasets[i]['train']['label'] > 0))
    train_neg.append(torch.count_nonzero(datasets[i]['train']['label'] < 0))
    train_zero.append(train[i] - (train_pos[i] + train_neg[i]))
    val.append(len(datasets[i]['val']['label'].squeeze()))
    val_pos.append(torch.count_nonzero(datasets[i]['val']['label'] > 0))
    val_neg.append(torch.count_nonzero(datasets[i]['val']['label'] < 0))
    val_zero.append(val[i] - (val_pos[i] + val_neg[i]))
    test.append(len(datasets[i]['test']['label'].squeeze()))
    test_pos.append(torch.count_nonzero(datasets[i]['test']['label'] > 0))
    test_neg.append(torch.count_nonzero(datasets[i]['test']['label'] < 0))
    test_zero.append(test[i] - (test_pos[i] + test_neg[i]))

print("Train: ", sum(train)/10)
print("Train pos: ", (sum(train_pos)/10).item())
print("Train neg: ", (sum(train_neg)/10).item())
print("Train zero: ", (sum(train_zero)/10).item())
print("Val: ", sum(val)/10)
print("Val pos: ", (sum(val_pos)/10).item())
print("Val neg: ", (sum(val_neg)/10).item())
print("Val zero: ", (sum(val_zero)/10).item())
print("Test: ", sum(test)/10)
print("Test pos: ", (sum(test_pos)/10).item())
print("Test neg: ", (sum(test_neg)/10).item())
print("Test zero: ", (sum(test_zero)/10).item())


