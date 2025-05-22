import numpy as np
import torch
import torch.nn
import math
import statistics
import random
from torch_geometric_signed_directed.data import load_directed_real_data
from pathlib import Path
import utils
import time

from msgnn import MSGNN_node_classification

randomseed = 0
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.use_deterministic_algorithms(True)

dataset_name = 'telegram' 
task = 'node_classification'
dropout = 0.5
normalize = True
epochs = 1000
early_stopping_threshold = 200

if dataset_name in ['telegram']:
    data = load_directed_real_data(dataset=dataset_name, root='./tmp/telegram/', name=dataset_name, data_split=10)
    if normalize == True:
        data.edge_weight = torch.exp(-1/data.edge_weight)

log_path = './tmp/res/Node_Msgnn_' + dataset_name
if normalize is True:
    log_path += '_normalized'
Path(log_path).mkdir(parents=True, exist_ok=True)

dataset = data

size = dataset.y.size(-1)
f_node, e_node = dataset.edge_index[0], dataset.edge_index[1]
X = dataset.x.data.numpy().astype('float32')
#X = utils.in_out_degree(dataset.edge_index, size, dataset.edge_weight)
label = dataset.y.data.numpy().astype('int')

train_mask = dataset.train_mask.data.numpy().astype('bool_')
val_mask = dataset.val_mask.data.numpy().astype('bool_')
test_mask = dataset.test_mask.data.numpy().astype('bool_')

# normalize label, the minimum should be 0 as class index
_label_ = label - np.amin(label)
label_dim = np.amax(_label_)+1

label = torch.from_numpy(_label_[np.newaxis])
X_img  = torch.FloatTensor(X)
X_real = torch.FloatTensor(X)

splits = train_mask.shape[1]
assert splits == 10

if len(test_mask.shape) == 1:
    test_mask = np.repeat(test_mask[:,np.newaxis], splits, 1)

nb_epochs = [0 for _ in range(360)]
it_epochs = -1

for lr in [0.001, 0.005, 0.01, 0.05]:
    for num_filter in [16, 32, 64]:
        for layer in [2, 4, 8]:
            current_params = 'lr_' + str(lr) + '_num_filter_' + str(num_filter) + '_layer_' + str(layer)
            print(current_params)

            torch.manual_seed(randomseed)
            
            for i in range(splits):
                it_epochs += 1
                log_str_full = ''

                model = MSGNN_node_classification(K=1, num_features=X_real.size(-1), hidden=num_filter, label_dim=label_dim, \
                            trainable_q = False, layer=layer, dropout=dropout, normalization='sym', cached=False)  
 
                opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

                #################################
                # Train/Validation/Test
                #################################
                
                train_index = train_mask[:, i]
                val_index = val_mask[:, i]
                test_index = test_mask[:, i]

                y_train = label[:,train_index]
                y_val   = label[:,val_index]
                y_test  = label[:,test_index]

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
                    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
                    pred_label = out.max(dim = 1)[1]
                    train_loss = torch.nn.functional.nll_loss(out[:,:,train_index], y_train)
                    train_acc  = utils.acc(pred_label[:,train_index].squeeze(), y_train)
                    opt.zero_grad()
                    train_loss.backward()
                    opt.step()

                    ####################
                    # Validation
                    ####################
                    val_loss, val_acc = 0.0, 0.0
                    model.eval()
                    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
                    pred_label = out.max(dim = 1)[1]
                    val_loss = torch.nn.functional.nll_loss(out[:,:,val_index], y_val)
                    val_acc = utils.acc(pred_label[:,val_index].squeeze(), y_val)

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

                val_index = val_mask[:, i]
                y_val   = label[:,val_index]

                model = MSGNN_node_classification(K=1, num_features=X_real.size(-1), hidden=num_filter, label_dim=label_dim, \
                            trainable_q = False, layer=layer, dropout=dropout, normalization='sym', cached=False)  
                model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+current_params+'.t7'))
                model.eval()
                out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
                pred_label = out.max(dim = 1)[1]
                i_validation_error_model_loss[i] = torch.nn.functional.nll_loss(out[:,:,val_index], y_val).detach().item()
                i_validation_error_model_acc[i] = utils.acc(pred_label[:, val_index].squeeze(), y_val)
                
                model = MSGNN_node_classification(K=1, num_features=X_real.size(-1), hidden=num_filter, label_dim=label_dim, \
                            trainable_q = False, layer=layer, dropout=dropout, normalization='sym', cached=False)  
                model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+current_params+'.t7'))
                model.eval()
                out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
                pred_label = out.max(dim = 1)[1]
                i_validation_acc_model_loss[i] = torch.nn.functional.nll_loss(out[:,:,val_index], y_val).detach().item()
                i_validation_acc_model_acc[i] = utils.acc(pred_label[:, val_index].squeeze(), y_val)


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

    val_index = val_mask[:, i]
    test_index = test_mask[:, i]

    y_val   = label[:,val_index]
    y_test  = label[:,test_index]

    model = MSGNN_node_classification(K=1, num_features=X_real.size(-1), hidden=best_error_model_num_filter, label_dim=label_dim, \
                            trainable_q = False, layer=best_error_model_layer, dropout=dropout, normalization='sym', cached=False)
    model.load_state_dict(torch.load(log_path + '/model_err'+str(i)+best_error_model_params+'.t7'))
    model.eval()
    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
    pred_label = out.max(dim = 1)[1]
    val_loss_err = torch.nn.functional.nll_loss(out[:,:,val_index], y_val).detach().item()
    val_acc_err = utils.acc(pred_label[:,val_index].squeeze(), y_val)
    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
    pred_label = out.max(dim = 1)[1]
    test_loss_err = torch.nn.functional.nll_loss(out[:,:,test_index], y_test).detach().item()
    t_errm_err[i] = test_loss_err
    test_acc_err = utils.acc(pred_label[:,test_index].squeeze(), y_test)
    t_errm_acc[i] = test_acc_err
    log_str = ('val_acc_err: {val_acc_err:.10f}, '+'val_loss_err: {val_loss_err:.10f}, '+'test_acc_err: {test_acc_err:.10f}, '+'test_loss_err: {test_loss_err:.10f} ,')
    log_testing_err_overall[i] += best_error_model_params + '\n'
    log_testing_err_overall[i] += log_str.format(val_acc_err = val_acc_err, val_loss_err = math.sqrt(val_loss_err), test_acc_err = test_acc_err, test_loss_err = math.sqrt(test_loss_err))
    log_testing_err_overall[i] += '\nepochs: '
    log_testing_err_overall[i] += str(nb_epochs[best_err_model_epochs_idx + i] + 1)
    with open(log_path + '/log_testing_err_overall'+str(i)+'.csv', 'w') as file:
        file.write(log_testing_err_overall[i])
        file.write('\n')

    model = MSGNN_node_classification(K=1, num_features=X_real.size(-1), hidden=best_acc_model_num_filter, label_dim=label_dim, \
                            trainable_q = False, layer=best_acc_model_layer, dropout=dropout, normalization='sym', cached=False)
    model.load_state_dict(torch.load(log_path + '/model_acc'+str(i)+best_acc_model_params+'.t7'))
    model.eval()
    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
    pred_label = out.max(dim = 1)[1]
    val_loss_acc = torch.nn.functional.nll_loss(out[:,:,val_index], y_val).detach().item()
    val_acc_acc = utils.acc(pred_label[:,val_index].squeeze(), y_val)
    out = model(X_real, X_img, dataset.edge_index, dataset.edge_weight)
    pred_label = out.max(dim = 1)[1]
    test_loss_acc = torch.nn.functional.nll_loss(out[:,:,test_index], y_test).detach().item()
    t_accm_err[i] = test_loss_acc
    test_acc_acc = utils.acc(pred_label[:,test_index].squeeze(), y_test)
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