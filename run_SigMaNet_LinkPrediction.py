import numpy as np
import random
import torch
import time
from torch_geometric_signed_directed.data import load_directed_real_data
import pickle as pk
import math
import utils
from pathlib import Path
from sklearn import preprocessing
import statistics
import read_datasets_additional

from sigmanet import SigMaNet_link_prediction_one_laplacian

randomseed = 0
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.use_deterministic_algorithms(True)

dataset_name = 'telegram' # ['telegram', 'bitcoin_alpha', 'bitcoin_otc', 'bitcoin_alpha+', 'bitcoin_otc+', 'ucsocial'] + synth_datasets
task = 'existence' # ['existence', 'three_class_digraph', 'weight_prediction']
dropout = 0.5
normalize = True
epochs = 1000
early_stopping_threshold = 200

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
    data = load_directed_real_data(dataset=dataset_name, root='./tmp/telegram/', name=dataset_name)
    if normalize == True:
        data.edge_weight = torch.exp(-1/data.edge_weight)
elif dataset_name in ['bitcoin_alpha', 'bitcoin_otc', 'bitcoin_alpha+', 'bitcoin_otc+', 'ucsocial']:
    if dataset_name == 'bitcoin_alpha' or dataset_name == 'bitcoin_otc':
        data = utils.load_signed_real_data_no_negative(dataset=dataset_name, root='./tmp/', keep_negatives=True)
    elif dataset_name == 'bitcoin_alpha+' or dataset_name == 'bitcoin_otc+':
        data = utils.load_signed_real_data_no_negative(dataset=dataset_name[:-1], root='./tmp/', keep_negatives=False)
    elif dataset_name == 'ucsocial':
        data = read_datasets_additional.ReadDataset(root='./tmp/', name=dataset_name)._data
    if normalize == True:
        if dataset_name != 'ucsocial':
            data.edge_weight = torch.tensor(preprocessing.MaxAbsScaler().fit_transform(data.edge_weight.reshape(-1, 1))).T.squeeze()
        else:
            data.edge_weight = torch.exp(-1/data.edge_weight)
else:
    synth_filename = synth_filenames[synth_datasets.index(dataset_name)]
    try:
        data = pk.load(open(f'./tmp/synthetic/{synth_filename}.pk','rb'))
        if normalize == True:
            data.edge_weight = torch.tensor(preprocessing.MaxAbsScaler().fit_transform((data.edge_weight.reshape(-1, 1)))).T.squeeze()
    except:
        raise Exception("Wrong dataset.")

log_path = './tmp/res/Edge_SigMaNet_' + dataset_name
if normalize is True:
    log_path += '_normalized'
Path(log_path).mkdir(parents=True, exist_ok=True)

edge_index = data.edge_index
size = torch.max(edge_index).item() + 1
data.num_nodes = size
datasets = utils.link_class_split_new(data, prob_val=0.05, prob_test=0.15, splits=10, task=task, \
                                      maintain_connect=True, keep_negatives=dataset_name != 'bitcoin_alpha+' and dataset_name != 'bitcoin_otc+', \
                                      seed=randomseed)

if task == 'weight_prediction':
    label_dim = 1
elif task == 'existence' :
    label_dim = 2
elif task == 'three_class_digraph':
    label_dim = 3
else:
    raise Exception("Wrong task.")

nb_epochs = [0 for _ in range(360)]
it_epochs = -1

for lr in [0.001, 0.005, 0.01, 0.05]:
    for num_filter in [16, 32, 64]:
        for layer in [2, 4, 8]:
            current_params = 'lr_' + str(lr) + '_num_filter_' + str(num_filter) + '_layer_' + str(layer)
            print(current_params)

            torch.manual_seed(randomseed)

            for i in range(10):
                it_epochs += 1
                log_str_full = ''
                ########################################
                # get hermitian laplacian
                ########################################
                edges = datasets[i]['graph']
                f_node, e_node = edges[0], edges[1]

                X_real = utils.in_out_degree(edges, size, datasets[i]['weights'])
                X_img = X_real.clone()
                edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                                    x_real=X_real, edge_weight=datasets[i]['weights'], normalization='sym', return_lambda_max=False)
                ########################################
                # initialize model and load dataset
                ########################################
                model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=label_dim, dropout=dropout,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag, weight_prediction=task == 'weight_prediction')

                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

                y_train = datasets[i]['train']['label']
                y_val   = datasets[i]['val']['label']
                y_test  = datasets[i]['test']['label']
                if task == 'weight_prediction':
                    y_train = y_train.float()
                    y_val   = y_val.float()
                    y_test  = y_test.float()
                else:
                    y_train = y_train.long()
                    y_val   = y_val.long()
                    y_test  = y_test.long()
                train_index = datasets[i]['train']['edges']
                val_index = datasets[i]['val']['edges']
                test_index = datasets[i]['test']['edges']

                #################################
                # Train/Validation/Test
                #################################
                best_test_err = 100000000000.0
                best_test_acc = -100000000000.0
                early_stopping = 0
                for epoch in range(epochs):
                    start_time = time.time()
                    if early_stopping > early_stopping_threshold:
                        break
                    nb_epochs[it_epochs] = epoch
                    ####################
                    # Train
                    ####################
                    train_loss, train_acc = 0.0, 0.0
                    model.train()
                    out = model(X_real, X_img, train_index)
                    if task == 'weight_prediction':
                        pred_label = out.T
                        train_loss = torch.nn.functional.mse_loss(out.T, y_train)
                    else:
                        pred_label = out.max(dim = 1)[1]
                        train_loss = torch.nn.functional.nll_loss(out, y_train)
                    train_acc  = utils.acc(pred_label, y_train, weighted=task == 'weight_prediction')
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                    ####################
                    # Validation
                    ####################
                    val_loss, val_acc = 0.0, 0.0
                    model.eval()
                    out = model(X_real, X_img, val_index)
                    if task == 'weight_prediction':
                        pred_label = out.T
                        val_loss = torch.nn.functional.mse_loss(out.T, y_val)
                    else:
                        pred_label = out.max(dim = 1)[1]  
                        val_loss = torch.nn.functional.nll_loss(out, y_val)   
                    val_acc = utils.acc(pred_label, y_val, weighted=task == 'weight_prediction')

                    ####################
                    # Save weights
                    ####################
                    save_perform_err = val_loss.detach().item()
                    save_perform_acc = val_acc
                    if save_perform_err <= best_test_err:
                        early_stopping = 0
                        best_test_err = save_perform_err
                        torch.save(model.state_dict(), log_path + '/model_err'+str(i)+current_params+'.t7')
                    if save_perform_acc >= best_test_acc:
                        #early_stopping = 0
                        best_test_acc = save_perform_acc
                        torch.save(model.state_dict(), log_path + '/model_acc'+str(i)+current_params+'.t7')
                    else:
                        early_stopping += 1
            torch.cuda.empty_cache()

err_model_best_average_loss = 100000000000.0
best_error_model_params = ''
best_error_model_num_filter = 0
best_error_model_layer = 0
best_error_model_lr = 0
best_error_model_log = ''

acc_model_best_average_acc = -100000000000.0
best_acc_model_params = ''
best_acc_model_num_filter = 0
best_acc_model_layer = 0
best_acc_model_lr = 0
best_acc_model_log = ''

it_epochs = -1
best_err_model_epochs_idx = -1
best_acc_model_epochs_idx = -1
for lr in [0.001, 0.005, 0.01, 0.05]:
    for num_filter in [16, 32, 64]:
        for layer in [2, 4, 8]:
            current_params = 'lr_' + str(lr) + '_num_filter_' + str(num_filter) + '_layer_' + str(layer)

            i_validation_error_model_acc = [0.0 for _ in range(10)]
            i_validation_error_model_loss = [0.0 for _ in range(10)]
            i_validation_acc_model_acc = [0.0 for _ in range(10)]
            i_validation_acc_model_loss = [0.0 for _ in range(10)]
            for i in range(10):
                it_epochs += 1
                edges = datasets[i]['graph']
                f_node, e_node = edges[0], edges[1]
                    
                y_val   = datasets[i]['val']['label']
                if task == 'weight_prediction':
                    y_val = y_val.float()
                else:
                    y_val = y_val.long()
                val_index = datasets[i]['val']['edges']

                X_real = utils.in_out_degree(edges, size, datasets[i]['weights'])
                X_img = X_real.clone()
                edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                    x_real=X_real, edge_weight=datasets[i]['weights'], normalization = 'sym', return_lambda_max = False)
                
                model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=label_dim, dropout=dropout,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag, weight_prediction=task == 'weight_prediction')

                model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+current_params+'.t7'))
                model.eval()
                out = model(X_real, X_img, val_index)
                if task == 'weight_prediction':
                    pred_label = out.T
                    i_validation_error_model_loss[i] = torch.nn.functional.mse_loss(out.T, y_val).detach().item()
                else:
                    pred_label = out.max(dim = 1)[1]
                    i_validation_error_model_loss[i] = torch.nn.functional.nll_loss(out, y_val).detach().item()

                i_validation_error_model_acc[i] = utils.acc(pred_label, y_val, weighted=task == 'weight_prediction')
                

                model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=num_filter, label_dim=label_dim, dropout=dropout,\
                            i_complex = False,  layer=layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                            norm_real=norm_real, norm_imag=norm_imag, weight_prediction=task == 'weight_prediction')

                model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+current_params+'.t7'))
                model.eval()
                out = model(X_real, X_img, val_index)
                if task == 'weight_prediction':
                    pred_label = out.T
                    i_validation_acc_model_loss[i] = torch.nn.functional.mse_loss(out.T, y_val).detach().item()
                else:
                    pred_label = out.max(dim = 1)[1]
                    i_validation_acc_model_loss[i] = torch.nn.functional.nll_loss(out, y_val).detach().item()

                i_validation_acc_model_acc[i] = utils.acc(pred_label, y_val, weighted=task == 'weight_prediction')


            if sum(i_validation_error_model_loss) / 10 < err_model_best_average_loss:
                best_err_model_epochs_idx = it_epochs - 9
                err_model_best_average_loss = sum(i_validation_error_model_loss) / 10
                best_error_model_params = current_params
                best_error_model_num_filter = num_filter
                best_error_model_layer = layer
                best_error_model_lr = lr
                best_error_model_log += 'i-th split validation loss: '
                for i in range(10):
                    log = ('{i}: {val_err_loss:.10f}')
                    log = log.format(i=i, val_err_loss=i_validation_error_model_loss[i])
                    best_error_model_log += log + ' '
                best_error_model_log += '\n i-th split validation acc:  '
                for i in range(10):
                    log = ('{i}: {val_err_acc:.10f}')
                    log = log.format(i=i, val_err_acc=i_validation_error_model_acc[i])
                    best_error_model_log += log + ' '
                avg_acc_err_model = sum(i_validation_error_model_acc) / 10
                best_error_model_log += '\nwith average loss ' + str(err_model_best_average_loss) + ' and average acc ' + str(avg_acc_err_model)
                best_error_model_log += '\n##########################################################################################\n'
            if sum(i_validation_acc_model_acc) / 10 > acc_model_best_average_acc:
                best_acc_model_epochs_idx = it_epochs - 9
                acc_model_best_average_acc = sum(i_validation_acc_model_acc) / 10
                best_acc_model_params = current_params
                best_acc_model_num_filter = num_filter
                best_acc_model_layer = layer
                best_acc_model_lr = lr
                best_acc_model_log += 'i-th split validation loss: '
                for i in range(10):
                    log = ('{i}: {val_acc_loss:.10f}')
                    log = log.format(i=i, val_acc_loss=i_validation_acc_model_loss[i])
                    best_acc_model_log += log + ' '
                best_acc_model_log += '\ni-th split validation acc:  '
                for i in range(10):
                    log = ('{i}: {val_acc_acc:.10f}')
                    log = log.format(i=i, val_acc_acc=i_validation_acc_model_acc[i])
                    best_acc_model_log += log + ' '
                avg_loss_acc_model = sum(i_validation_acc_model_loss) / 10
                best_acc_model_log += '\nwith average loss ' + str(avg_loss_acc_model) + ' and average acc ' + str(acc_model_best_average_acc)
                best_acc_model_log += '\n##########################################################################################\n'

with open(log_path + '/best_error_model_validation_search_log'+'.csv', 'w') as file:
    file.write(best_error_model_log)
    file.write('\n')
with open(log_path + '/best_acc_model_validation_search_log'+'.csv', 'w') as file:
    file.write(best_acc_model_log)
    file.write('\n')

t_errm_acc = [0 for _ in range(10)]
t_errm_err = [0 for _ in range(10)]
t_accm_acc = [0 for _ in range(10)]
t_accm_err = [0 for _ in range(10)]

log_testing_err_overall = ['' for _ in range(10)]
log_testing_acc_overall = ['' for _ in range(10)]
for i in range(10):
    edges = datasets[i]['graph']
    f_node, e_node = edges[0], edges[1]
                        
    X_real = utils.in_out_degree(edges, size, datasets[i]['weights'])
    X_img = X_real.clone()

    edge_index, norm_real, norm_imag = utils.process_magnetic_laplacian(edge_index=edges, gcn=True, net_flow=True,\
                                        x_real=X_real, edge_weight=datasets[i]['weights'], normalization = 'sym', return_lambda_max = False)                  

    y_val   = datasets[i]['val']['label']
    y_test  = datasets[i]['test']['label']
    if task == 'weight_prediction':
        y_val   = y_val.float()
        y_test  = y_test.float()
    else:
        y_val = y_val.long()
        y_test = y_test.long()

    val_index = datasets[i]['val']['edges']
    test_index = datasets[i]['test']['edges']

    model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=best_error_model_num_filter, label_dim=label_dim, dropout=dropout,\
                i_complex = False,  layer=best_error_model_layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                norm_real=norm_real, norm_imag=norm_imag, weight_prediction=task == 'weight_prediction')
    model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+best_error_model_params+'.t7'))
    model.eval()
    out = model(X_real, X_img, val_index)
    if task == 'weight_prediction':
        pred_label = out.T
        val_loss_err = torch.nn.functional.mse_loss(out.T, y_val).detach().item()
    else:
        pred_label = out.max(dim = 1)[1]
        val_loss_err = torch.nn.functional.nll_loss(out, y_val).detach().item()
    val_acc_err = utils.acc(pred_label, y_val, weighted=task == 'weight_prediction')
    out = model(X_real, X_img, test_index)
    if task == 'weight_prediction':
        pred_label = out.T
        test_loss_err = torch.nn.functional.mse_loss(out.T, y_test).detach().item()
        t_errm_err[i] = test_loss_err
    else:
        pred_label = out.max(dim = 1)[1]
        test_loss_err = torch.nn.functional.nll_loss(out, y_test).detach().item()
        t_errm_err[i] = test_loss_err
    test_acc_err = utils.acc(pred_label, y_test, weighted=task == 'weight_prediction')
    t_errm_acc[i] = test_acc_err
    log_str = ('val_acc_err: {val_acc_err:.10f}, '+'val_loss_err: {val_loss_err:.10f}, '+'test_acc_err: {test_acc_err:.10f}, '+'test_loss_err: {test_loss_err:.10f} ,')
    log_testing_err_overall[i] += best_error_model_params + '\n'
    log_testing_err_overall[i] += log_str.format(val_acc_err = val_acc_err, val_loss_err = math.sqrt(val_loss_err), test_acc_err = test_acc_err, test_loss_err = math.sqrt(test_loss_err))
    log_testing_err_overall[i] += '\nepochs: '
    log_testing_err_overall[i] += str(nb_epochs[best_err_model_epochs_idx + i] + 1)
    with open(log_path + '/log_testing_err_overall'+str(i)+'.csv', 'w') as file:
        file.write(log_testing_err_overall[i])
        file.write('\n')

    model = SigMaNet_link_prediction_one_laplacian(K=1, num_features=2, hidden=best_acc_model_num_filter, label_dim=label_dim, dropout=dropout,\
                i_complex = False,  layer=best_acc_model_layer, follow_math=True, gcn=True, net_flow=True, unwind=True, edge_index=edge_index,\
                norm_real=norm_real, norm_imag=norm_imag, weight_prediction=task == 'weight_prediction')
    model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+best_acc_model_params+'.t7'))
    model.eval()
    out = model(X_real, X_img, val_index)
    if task == 'weight_prediction':
        pred_label = out.T
        val_loss_acc = torch.nn.functional.mse_loss(out.T, y_val).detach().item()
    else:
        pred_label = out.max(dim = 1)[1]
        val_loss_acc = torch.nn.functional.nll_loss(out, y_val).detach().item()
    val_acc_acc = utils.acc(pred_label, y_val, weighted=task == 'weight_prediction')
    out = model(X_real, X_img, test_index)
    if task == 'weight_prediction':
        pred_label = out.T
        test_loss_acc = torch.nn.functional.mse_loss(out.T, y_test).detach().item()
        t_accm_err[i] = test_loss_acc
    else:
        pred_label = out.max(dim = 1)[1]
        test_loss_acc = torch.nn.functional.nll_loss(out, y_test).detach().item()
        t_accm_err[i] = test_loss_acc

    test_acc_acc = utils.acc(pred_label, y_test, weighted=task == 'weight_prediction')
    t_accm_acc[i] = test_acc_acc
    log_str = ('val_acc: {val_acc_acc:.10f}, '+'val_loss: {val_loss_acc:.10f}, '+'test_acc: {test_acc_acc:.10f}, '+'test_loss: {test_loss_acc:.10f}, ')
    log_testing_acc_overall[i] += best_acc_model_params + '\n'
    log_testing_acc_overall[i] += log_str.format(val_acc_acc = val_acc_acc, val_loss_acc = math.sqrt(val_loss_acc), test_acc_acc = test_acc_acc, test_loss_acc = math.sqrt(test_loss_acc))
    log_testing_acc_overall[i] += '\nepochs: '
    log_testing_acc_overall[i] += str(nb_epochs[best_acc_model_epochs_idx + i] + 1)
    with open(log_path + '/log_testing_acc_overall'+str(i)+'.csv', 'w') as file:
        file.write(log_testing_acc_overall[i])
        file.write('\n')

with open (log_path + '/results.csv', 'w') as file:
    file.write('acc models mean acc: ' + str(statistics.mean(t_accm_acc)) + ' +- ' + str(statistics.pstdev(t_accm_acc)))
    file.write('\n')
    file.write('acc models mean loss:' + str(statistics.mean([math.sqrt(x) for x in t_accm_err])) + ' +- ' + str(statistics.pstdev([math.sqrt(x) for x in t_accm_err])))
    file.write('\n')
    file.write('err models mean acc: ' + str(statistics.mean(t_errm_acc)) + ' +- ' + str(statistics.pstdev(t_errm_acc)))
    file.write('\n')
    file.write('err models mean loss: ' + str(statistics.mean([math.sqrt(x) for x in t_errm_err])) + ' +- ' + str(statistics.pstdev([math.sqrt(x) for x in t_errm_err])))
    file.write('\n')