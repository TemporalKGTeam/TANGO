import numpy as np
import time
import os
import torch
import argparse

from models.MGCN import *
from TANGO_dataloader import *
from models.models import TANGO
from utils import *
from eval import *

def save_model(model, args, best_val, best_epoch, optimizer, save_path):

    state = {
        'state_dict': model.state_dict(),
        'best_val': best_val,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict(),
        'args'	: vars(args)
    }
    torch.save(state, save_path)

def load_model(load_path, optimizer, model):
    state = torch.load(load_path, map_location={'cuda:3': 'cuda:1'})
    state_dict = state['state_dict']
    best_val = state['best_val']
    best_val_mrr = best_val['mrr']

    model.load_state_dict(state_dict)
    optimizer.load_state_dict(state['optimizer'])

    return best_val_mrr

def load_emb(load_path, model):
    state = torch.load(load_path, map_location={'cuda:3': 'cuda:1'})
    state_dict = state['state_dict']
    model.load_state_dict(state_dict)

def adjust_learning_rate(optimizer, lr, gamma):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr_ = lr * gamma
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_
    return lr_

if __name__ == '__main__':
    modelpth = './checkpoints/'
    parser = argparse.ArgumentParser(description='TANGO Training Parameters')
    parser.add_argument('--gde_core', type=str, default='mgcn', help='core layer function of the TANGO model')
    parser.add_argument('--score_func', type=str, default='tucker', help='score function')
    parser.add_argument('--core_layer', type=int, default=2, help='number of core function layers')
    parser.add_argument('--num_epoch', type=int, default=100, help='number of maximum epoch')
    parser.add_argument('--test_step', type=int, default=1, help='number of epochs after which we do evaluation')
    parser.add_argument('--input_step', type=int, default=4, help='number of input steps for ODEblock')
    parser.add_argument('--delta_step', type=int, default=0, help='number of steps between the last input snapshot and the prediction snapshot')
    parser.add_argument('--target_step', type=int, default=1, help='number of prediction snapshots')
    parser.add_argument('--initsize', type=int, default=200, help='size of initial representation dimension')
    parser.add_argument('--embsize', type=int, default=200, help='size of output embeddings')
    parser.add_argument('--hidsize', type=int, default= 200, help='size of representation dimension in the core function')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--solver', type=str, default='rk4', help='ODE solver')
    parser.add_argument('--atol', type=float, default='1e-4', help='lower bound of the tolerance')
    parser.add_argument('--rtol', type=float, default='1e-3', help='higher bound of the tolerance')
    parser.add_argument('--device', type=str, default='cuda:0', help='device name')
    parser.add_argument('--dataset', type=str, default='ICEWS05-15', help='dataset name')
    parser.add_argument('--scale', type=float, default=0.1, help='scale the length of integration')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout')
    parser.add_argument('--bias', action='store_false', help='whether to use bias in relation specific transformation')
    parser.add_argument('--adjoint_flag', action='store_false', help='whether to use adjoint method')
    parser.add_argument('--opn', type=str, default='mult', help='composition operation to be used in MGCN')
    parser.add_argument('--shuffle', action='store_false', help='shuffle in dataloader')
    parser.add_argument('--cheby_grid', type=int, default=3, help='number of chebyshev nodes, without chebyshev approximation if cheby_grid=0')
    parser.add_argument('--resume', action='store_true', help='retore a model')
    parser.add_argument('--name', type=str, default='TANGO', help='name of the run')
    parser.add_argument('--jump', action='store_true', help='whether to use graph transition layer')
    parser.add_argument('--jump_init', type=float, default=0.01, help='weight of transition term')
    parser.add_argument('--activation', type=str, default='relu', help='activation function')
    parser.add_argument('--res', action='store_true', help='include residual MGCN layer')
    parser.add_argument('--rel_jump', action='store_true', help='include transition tensor')
    parser.add_argument('--induct_test', action='store_true', help='inductive link prediction')
    parser.add_argument('--test', action='store_true', help='store to start the test, otherwise start training')

    args = parser.parse_args()
    if not args.resume: args.name = args.name + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H:%M:%S')

    logger = setup_logger(args.name)


    if not os.path.exists(modelpth):
        os.mkdir(modelpth)

    loadpth = modelpth + args.name
    device = args.device if torch.cuda.is_available() else 'cpu'
    args.device = device

    print("Using device: ", device)
    logger.info(vars(args))

    # seed for repeatability
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(0)
    np.random.seed(0)

    if args.dataset == 'ICEWS14':
        val_exist = 0 # ICEWS14 does not have validation set
    else:
        val_exist = 1

    if val_exist:
        if args.induct_test:
            num_e, num_rel, test_timestamps, test_adj, test_triple, test_1nei, t_indep_trp, test_so2r, induct_tar = setup_induct_test(
                args.dataset, logger, args.scale, args.input_step)
            adjlist = load_adjmtx(args.dataset)
            test_adjmtx = adjlist[-len(test_timestamps):]
        else:
            induct_tar = None

            num_e, num_rel, train_timestamps, test_timestamps, val_timestamps, train_adj, test_adj, val_adj, train_triple, test_triple, val_triple, \
            train_1nei, test_1nei, val_1nei, t_indep_trp, train_so2r, val_so2r, test_so2r = setup_tKG(args.dataset,
                                                                                                      logger,
                                                                                                      args.initsize,
                                                                                                      args.scale,
                                                                                                      val_exist,
                                                                                                      args.input_step)
            trainl, testl, vall = len(train_timestamps), len(test_timestamps)-args.input_step, len(val_timestamps)-args.input_step
            adjlist = load_adjmtx(args.dataset)
            train_adjmtx, test_adjmtx, val_adjmtx = adjlist[:trainl], adjlist[trainl+vall-args.input_step:], adjlist[trainl-args.input_step:trainl+vall]

    else:
        induct_tar = None

        num_e, num_rel, train_timestamps, test_timestamps, train_adj, test_adj, train_triple, test_triple, train_1nei, test_1nei, t_indep_trp, train_so2r, test_so2r \
            = setup_tKG(args.dataset, logger, args.initsize, args.scale, val_exist, args.input_step)
        trainl, testl = len(train_timestamps), len(test_timestamps)-args.input_step
        adjlist = load_adjmtx(args.dataset)
        train_adjmtx, test_adjmtx = adjlist[:trainl], adjlist[trainl-args.input_step:trainl+testl]


    if args.induct_test:
        test_dataset = TANGOtestDataset(args,
                                         test_triple,
                                         test_adj,
                                         test_adjmtx,
                                         test_so2r,
                                         num_e,
                                         input_steps=args.input_step,
                                         target_steps=args.target_step,
                                         delta_steps=args.delta_step,
                                         time_stamps=test_timestamps,
                                         t_indep_trp=t_indep_trp,
                                         induct_tar=induct_tar)
        test_loader = TANGOtestDataLoader(dataset=test_dataset,
                                           batch_size=1,
                                           shuffle=args.shuffle)
    else:
        train_dataset = TANGOtrainDataset(args,
                                           train_triple,
                                           train_adj,
                                           train_adjmtx,
                                           train_so2r,
                                           num_e,
                                           input_steps=args.input_step,
                                           target_steps=args.target_step,
                                           delta_steps=args.delta_step,
                                           time_stamps=train_timestamps,
                                           neg_samp=False)

        test_dataset = TANGOtestDataset(args,
                                         test_triple,
                                         test_adj,
                                         test_adjmtx,
                                         test_so2r,
                                         num_e,
                                         input_steps=args.input_step,
                                         target_steps=args.target_step,
                                         delta_steps=args.delta_step,
                                         time_stamps=test_timestamps,
                                         t_indep_trp=t_indep_trp)

        train_loader = TANGOtrainDataLoader(dataset=train_dataset,
                                  batch_size=1,
                                  shuffle=args.shuffle)

        test_loader = TANGOtestDataLoader(dataset=test_dataset,
                                 batch_size=1,
                                 shuffle=args.shuffle)

        if val_exist:
            val_dataset = TANGOtestDataset(args,
                                            val_triple,
                                            val_adj,
                                            val_adjmtx,
                                            val_so2r,
                                            num_e,
                                            input_steps=args.input_step,
                                            target_steps=args.target_step,
                                            delta_steps=args.delta_step,
                                            time_stamps=val_timestamps,
                                            t_indep_trp=t_indep_trp)

            val_loader = TANGOtestDataLoader(dataset=val_dataset,
                                    batch_size=1,
                                    shuffle=False)

            eval_loader = val_loader
        else:
            eval_loader = test_loader


    # instantiate model
    model = TANGO(num_e, num_rel, args, device, logger)
    model.to(device)

    for name, param in model.named_parameters():
        print(name, '     ', param.size())

    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr = args.lr

    best_val_mrr = 0
    # resume unfinished training process
    if args.resume:
        best_val_mrr = load_model(loadpth, optim, model)
        logger.info('Successfully Loaded previous model')

    kill_cnt = 0 # counts the number of epochs before early stop
    if args.induct_test == False and args.test == False:
        for epoch in range(args.num_epoch):
            running_loss = 0
            batch_num = 0
            ftime = 0 # forward time
            btime = 0 # backward time

            t1 = time.time()

            for step, (sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edge_idlist, \
                edge_typelist, adj_mtx, edge_jump_w, edge_jump_id, rel_jump) in enumerate(train_loader):
                optim.zero_grad()
                model.train()
                
                # forward
                t3 = time.time()
                loss = model(sub_tar, rel_tar, obj_tar, lab_tar, in_ts, tar_ts, edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump)
                t4 = time.time()
                ftime += (t4 - t3)

                # backward
                loss.backward()
                optim.step()
                t5 = time.time()
                btime += (t5 - t4)

                running_loss += loss.item()
                batch_num += 1

            running_loss /= batch_num # average loss
            t2 = time.time()

            # report loss information
            print("Epoch " + str(epoch + 1) + ": " + str(running_loss) + " Time: " + str(t2-t1))
            logger.info("Epoch " + str(epoch + 1) + ": " + str(running_loss) + " Time: " + str(t2-t1))
            # report forward and backward time
            print("Epoch " + str(epoch + 1) + ": Forward Time: " + str(ftime) + "     Backward Time: " + str(btime))
            logger.info("Epoch " + str(epoch + 1) + ": Forward Time: " + str(ftime) + "     Backward Time: " + str(btime))

            # evaluation
            if (epoch+1) % args.test_step == 0:
                if val_exist:
                    split = 'val'
                    results = predict(val_loader, model, args, num_e, test_adjmtx, logger)
                else:
                    split = 'test'
                    results = predict(test_loader, model, args, num_e, test_adjmtx, logger)

                print("===========RAW===========")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

                print("=====TIME AWARE FILTER=====")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

                print("====TIME UNAWARE FILTER====")
                print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
                print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
                print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
                print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
                print("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

                logger.info("===========RAW===========")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

                logger.info("=====TIME AWARE FILTER=====")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

                logger.info("====TIME UNAWARE FILTER====")
                logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
                logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
                logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
                logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
                logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

                if results['mrr'] > best_val_mrr:
                    # update best result
                    best_val = results
                    best_val_mrr = results['mrr']
                    best_epoch = epoch
                    save_model(model, args, best_val, best_epoch, optim, loadpth)
                    kill_cnt = 0
                else:
                    # early stop condition
                    kill_cnt += 1
                    if kill_cnt > 30:
                        logger.info("Early Stopping!!")
                        break

                print("========BEST MRR=========")
                print("Epoch {}, MRR {}".format(epoch + 1, best_val_mrr))
                logger.info("========BEST MRR=========")
                logger.info("Epoch {}, MRR {}".format(epoch + 1, best_val_mrr))

    else:
        if args.induct_test: # inductive link prediction, run if you have a trained model
            print("Start inductive testing...")
            logger.info("Start inductive testing...")
        else:
            print("Start testing...")
            logger.info("Start testing...")
        epoch = 0
        split = 'test'
        results = predict(test_loader, model, args, num_e, test_adjmtx, logger)

        print("===========RAW===========")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

        print("=====TIME AWARE FILTER=====")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

        print("====TIME UNAWARE FILTER====")
        print("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
        print("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
        print("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
        print("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
        print("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))

        logger.info("===========RAW===========")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_raw']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_raw']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_raw']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_raw']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_raw']))

        logger.info("=====TIME AWARE FILTER=====")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar']))

        logger.info("====TIME UNAWARE FILTER====")
        logger.info("Epoch {}, HITS10 {}".format(epoch + 1, results['hits@10_ind']))
        logger.info("Epoch {}, HITS3 {}".format(epoch + 1, results['hits@3_ind']))
        logger.info("Epoch {}, HITS1 {}".format(epoch + 1, results['hits@1_ind']))
        logger.info("Epoch {}, MRR {}".format(epoch + 1, results['mrr_ind']))
        logger.info("Epoch {}, MAR {}".format(epoch + 1, results['mar_ind']))