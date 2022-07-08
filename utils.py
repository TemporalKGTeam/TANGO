import torch
import pickle
import logging
import os


def setup_logger(name):
    cur_dir = os.getcwd()
    if not os.path.exists(cur_dir + "/log/"):
        os.mkdir(cur_dir + "/log/")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
        datefmt="%m-%d %H:%M",
        filename=cur_dir + "/log/" + name + ".log",
        filemode="a",
    )
    logger = logging.getLogger(name)
    return logger


def setup_tKG(dataset, logger, embsize, scale, val_exist, input_step):
    # load data after preprocess
    loaddata = open("{}/data_tKG.pkl".format(dataset), "rb")
    data = pickle.load(loaddata)
    loaddata.close()

    loadsr2o = open("{}/sr2o_all_tKG.pkl".format(dataset), "rb")
    sr2o = pickle.load(loadsr2o)
    loadsr2o.close()

    loadso2r = open("{}/so2r_all_tKG.pkl".format(dataset), "rb")
    so2r = pickle.load(loadso2r)
    loadso2r.close()

    loadtriples = open("{}/triples_tKG.pkl".format(dataset), "rb")
    triples = pickle.load(loadtriples)
    loadtriples.close()

    loadadjs = open("{}/adjs_tKG.pkl".format(dataset), "rb")
    adjs = pickle.load(loadadjs)
    loadadjs.close()

    loadtimestamp = open("{}/timestamp_tKG.pkl".format(dataset), "rb")
    timestamp = pickle.load(loadtimestamp)
    loadtimestamp.close()

    loadindep = open("{}/t_indep_trp.pkl".format(dataset), "rb")
    t_indep_trp = pickle.load(loadindep)
    loadindep.close()

    loadnei = open("{}/neighbor_tKG.pkl".format(dataset), "rb")
    neighbor = pickle.load(loadnei)
    loadnei.close()

    def get_total_number(inPath, fileName):
        with open(os.path.join(inPath, fileName), "r") as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    num_e, num_rel = get_total_number("{}/".format(dataset), "stat.txt")
    logger.info("number of entities:" + str(num_e))
    logger.info("number of relations:" + str(num_rel))

    if val_exist:
        # timestamps
        # normalize timestamps, and scale
        ts_max = max(
            max(max(timestamp["train"]), max(timestamp["test"])),
            max(timestamp["valid"]),
        )  # max timestamp in the dataset
        train_timestamps = (
            torch.tensor(timestamp["train"]) / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        test_timestamps = (
            torch.tensor(timestamp["test"]) / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        val_timestamps = (
            torch.tensor(timestamp["valid"]) / torch.tensor(ts_max, dtype=torch.float)
        ) * scale

        # extend val and test timestamps
        val_timestamps = torch.cat(
            [train_timestamps[-input_step:], val_timestamps], dim=0
        )
        test_timestamps = torch.cat(
            [val_timestamps[-input_step:], test_timestamps], dim=0
        )

        print("number of training snapshots:", len(timestamp["train"]))
        print("number of validation snapshots:", len(timestamp["valid"]))
        print("number of testing snapshots:", len(timestamp["test"]))
        logger.info("number of training snapshots:" + str(len(timestamp["train"])))
        logger.info("number of validation snapshots:" + str(len(timestamp["valid"])))
        logger.info("number of testing snapshots:" + str(len(timestamp["test"])))

        # adjs
        train_adj = adjs["train"]
        test_adj = adjs["test"]
        val_adj = adjs["valid"]

        # extend val and test adj
        val_adj_extend = train_adj[-input_step:]
        val_adj = val_adj_extend + val_adj
        test_adj_extend = val_adj[-input_step:]
        test_adj = test_adj_extend + test_adj

        # triples
        train_triple = triples["train"]
        val_triple = triples["valid"]
        test_triple = triples["test"]

        # extend val and test triples
        val_triple_extend = train_triple[-input_step:]
        val_triple = val_triple_extend + val_triple
        test_triple_extend = val_triple[-input_step:]
        test_triple = test_triple_extend + test_triple

        # one hop neighbor
        train_1nei = neighbor["train"]
        test_1nei = neighbor["test"]
        val_1nei = neighbor["valid"]

        # extend val and test neighbor
        val_1nei_extend = train_1nei[-input_step:]
        val_1nei = val_1nei_extend + val_1nei
        test_1nei_extend = val_1nei[-input_step:]
        test_1nei = test_1nei_extend + test_1nei

        # so2r
        train_so2r = so2r["train"]
        val_so2r = so2r["valid"]
        test_so2r = so2r["test"]

        # extend val and test so2r
        val_so2r_extend = train_so2r[-input_step:]
        val_so2r = val_so2r_extend + val_so2r
        test_so2r_extend = val_so2r[-input_step:]
        test_so2r = test_so2r_extend + test_so2r

    else:
        # timestamps
        # normalize timestamps, and scale
        ts_max = max(
            max(timestamp["train"]), max(timestamp["test"])
        )  # max timestamp in the dataset
        train_timestamps = (
            torch.tensor(timestamp["train"]) / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        test_timestamps = (
            torch.tensor(timestamp["test"]) / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        # train_timestamps = torch.tensor(timestamp['train']) * scale / 2.4
        # test_timestamps = torch.tensor(timestamp['test']) * scale / 2.4

        # extend test timestamps
        test_timestamps = torch.cat(
            [train_timestamps[-input_step:], test_timestamps], dim=0
        )

        print("number of training snapshots:", len(timestamp["train"]))
        print("number of testing snapshots:", len(timestamp["test"]))
        logger.info("number of training snapshots:" + str(len(timestamp["train"])))
        logger.info("number of testing snapshots:" + str(len(timestamp["test"])))

        # adjs
        train_adj = adjs["train"]
        test_adj = adjs["test"]

        # extend test adj
        test_adj_extend = train_adj[-input_step:]
        test_adj = test_adj_extend + test_adj

        # triples
        train_triple = triples["train"]
        test_triple = triples["test"]

        # extend test triples
        test_triple_extend = train_triple[-input_step:]
        test_triple = test_triple_extend + test_triple

        # one hop neighbor
        train_1nei = neighbor["train"]
        test_1nei = neighbor["test"]

        # extend test neighbor
        test_1nei_extend = train_1nei[-input_step:]
        test_1nei = test_1nei_extend + test_1nei

        # so2r
        train_so2r = so2r["train"]
        test_so2r = so2r["test"]

        # extend val and test so2r
        test_so2r_extend = train_so2r[-input_step:]
        test_so2r = test_so2r_extend + test_so2r

    if val_exist:
        return (
            num_e,
            num_rel,
            train_timestamps,
            test_timestamps,
            val_timestamps,
            train_adj,
            test_adj,
            val_adj,
            train_triple,
            test_triple,
            val_triple,
            train_1nei,
            test_1nei,
            val_1nei,
            t_indep_trp,
            train_so2r,
            val_so2r,
            test_so2r,
        )
        # return num_e, num_rel, train_node_feature, test_node_feature, val_node_feature, train_timestamps, test_timestamps, val_timestamps, train_adj, test_adj, val_adj, triples
    else:
        return (
            num_e,
            num_rel,
            train_timestamps,
            test_timestamps,
            train_adj,
            test_adj,
            train_triple,
            test_triple,
            train_1nei,
            test_1nei,
            t_indep_trp,
            train_so2r,
            test_so2r,
        )
        # return num_e, num_rel, train_node_feature, test_node_feature, train_timestamps, test_timestamps, train_adj, test_adj, triples


def setup_tKG2(dataset, logger, embsize, scale, val_exist, input_step):
    # load data after preprocess
    loaddata = open("{}/data_tKG_j.pkl".format(dataset), "rb")
    data = pickle.load(loaddata)
    loaddata.close()

    loadsr2o = open("{}/sr2o_all_tKG_j.pkl".format(dataset), "rb")
    sr2o = pickle.load(loadsr2o)
    loadsr2o.close()

    loadso2r = open("{}/so2r_all_tKG_j.pkl".format(dataset), "rb")
    so2r = pickle.load(loadso2r)
    loadso2r.close()

    loadtriples = open("{}/triples_tKG_j.pkl".format(dataset), "rb")
    triples = pickle.load(loadtriples)
    loadtriples.close()

    loadadjs = open("{}/adjs_tKG_j.pkl".format(dataset), "rb")
    adjs = pickle.load(loadadjs)
    loadadjs.close()

    loadtimestamp = open("{}/timestamp_tKG_j.pkl".format(dataset), "rb")
    timestamp = pickle.load(loadtimestamp)
    loadtimestamp.close()

    loadindep = open("{}/t_indep_trp_j.pkl".format(dataset), "rb")
    t_indep_trp = pickle.load(loadindep)
    loadindep.close()

    loadnei = open("{}/neighbor_tKG_j.pkl".format(dataset), "rb")
    neighbor = pickle.load(loadnei)
    loadnei.close()

    def get_total_number(inPath, fileName):
        with open(os.path.join(inPath, fileName), "r") as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    num_e, num_rel = get_total_number("{}/".format(dataset), "stat.txt")
    # num_rel *= 2 # include inv rel
    logger.info("number of entities:" + str(num_e))
    logger.info("number of relations:" + str(num_rel))

    if val_exist:
        # timestamps
        # normalize timestamps, and scale
        ts_max = max(
            max(max(timestamp["train_jump"]), max(timestamp["test_jump"])),
            max(timestamp["valid_jump"]),
        )  # max timestamp in the dataset
        train_timestamps = (
            torch.tensor(timestamp["train_jump"])
            / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        test_timestamps = (
            torch.tensor(timestamp["test_jump"])
            / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        val_timestamps = (
            torch.tensor(timestamp["valid_jump"])
            / torch.tensor(ts_max, dtype=torch.float)
        ) * scale

        # extend val and test timestamps
        val_timestamps = torch.cat(
            [train_timestamps[-input_step:], val_timestamps], dim=0
        )
        test_timestamps = torch.cat(
            [val_timestamps[-input_step:], test_timestamps], dim=0
        )

        print("number of training snapshots:", len(timestamp["train_jump"]))
        print("number of validation snapshots:", len(timestamp["valid_jump"]))
        print("number of testing snapshots:", len(timestamp["test_jump"]))
        logger.info("number of training snapshots:" + str(len(timestamp["train_jump"])))
        logger.info(
            "number of validation snapshots:" + str(len(timestamp["valid_jump"]))
        )
        logger.info("number of testing snapshots:" + str(len(timestamp["test_jump"])))

        # adjs
        train_adj = adjs["train_jump"]
        test_adj = adjs["test_jump"]
        val_adj = adjs["valid_jump"]

        # extend val and test adj
        val_adj_extend = train_adj[-input_step:]
        val_adj = val_adj_extend + val_adj
        test_adj_extend = val_adj[-input_step:]
        test_adj = test_adj_extend + test_adj

        # triples
        train_triple = triples["train_jump"]
        val_triple = triples["valid_jump"]
        test_triple = triples["test_jump"]

        # extend val and test triples
        val_triple_extend = train_triple[-input_step:]
        val_triple = val_triple_extend + val_triple
        test_triple_extend = val_triple[-input_step:]
        test_triple = test_triple_extend + test_triple

        # one hop neighbor
        train_1nei = neighbor["train_jump"]
        test_1nei = neighbor["test_jump"]
        val_1nei = neighbor["valid_jump"]

        # extend val and test neighbor
        val_1nei_extend = train_1nei[-input_step:]
        val_1nei = val_1nei_extend + val_1nei
        test_1nei_extend = val_1nei[-input_step:]
        test_1nei = test_1nei_extend + test_1nei

        # so2r
        train_so2r = so2r["train_jump"]
        val_so2r = so2r["valid_jump"]
        test_so2r = so2r["test_jump"]

        # extend val and test so2r
        val_so2r_extend = train_so2r[-input_step:]
        val_so2r = val_so2r_extend + val_so2r
        test_so2r_extend = val_so2r[-input_step:]
        test_so2r = test_so2r_extend + test_so2r

    else:
        # timestamps
        # normalize timestamps, and scale
        ts_max = max(
            max(timestamp["train"]), max(timestamp["test_jump"])
        )  # max timestamp in the dataset
        train_timestamps = (
            torch.tensor(timestamp["train_jump"])
            / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        test_timestamps = (
            torch.tensor(timestamp["test_jump"])
            / torch.tensor(ts_max, dtype=torch.float)
        ) * scale
        # train_timestamps = torch.tensor(timestamp['train']) * scale / 2.4
        # test_timestamps = torch.tensor(timestamp['test']) * scale / 2.4

        # extend test timestamps
        test_timestamps = torch.cat(
            [train_timestamps[-input_step:], test_timestamps], dim=0
        )

        print("number of training snapshots:", len(timestamp["train_jump"]))
        print("number of testing snapshots:", len(timestamp["test_jump"]))
        logger.info("number of training snapshots:" + str(len(timestamp["train_jump"])))
        logger.info("number of testing snapshots:" + str(len(timestamp["test_jump"])))

        # adjs
        train_adj = adjs["train_jump"]
        test_adj = adjs["test_jump"]

        # extend test adj
        test_adj_extend = train_adj[-input_step:]
        test_adj = test_adj_extend + test_adj

        # triples
        train_triple = triples["train_jump"]
        test_triple = triples["test_jump"]

        # extend test triples
        test_triple_extend = train_triple[-input_step:]
        test_triple = test_triple_extend + test_triple

        # one hop neighbor
        train_1nei = neighbor["train_jump"]
        test_1nei = neighbor["test_jump"]

        # extend test neighbor
        test_1nei_extend = train_1nei[-input_step:]
        test_1nei = test_1nei_extend + test_1nei

        # so2r
        train_so2r = so2r["train_jump"]
        test_so2r = so2r["test_jump"]

        # extend val and test so2r
        test_so2r_extend = train_so2r[-input_step:]
        test_so2r = test_so2r_extend + test_so2r

    if val_exist:
        return (
            num_e,
            num_rel,
            train_timestamps,
            test_timestamps,
            val_timestamps,
            train_adj,
            test_adj,
            val_adj,
            train_triple,
            test_triple,
            val_triple,
            train_1nei,
            test_1nei,
            val_1nei,
            t_indep_trp,
            train_so2r,
            val_so2r,
            test_so2r,
        )
        # return num_e, num_rel, train_node_feature, test_node_feature, val_node_feature, train_timestamps, test_timestamps, val_timestamps, train_adj, test_adj, val_adj, triples
    else:
        return (
            num_e,
            num_rel,
            train_timestamps,
            test_timestamps,
            train_adj,
            test_adj,
            train_triple,
            test_triple,
            train_1nei,
            test_1nei,
            t_indep_trp,
            train_so2r,
            test_so2r,
        )
        # return num_e, num_rel, train_node_feature, test_node_feature, train_timestamps, test_timestamps, train_adj, test_adj, triples


def setup_induct_test(dataset, logger, scale, input_step):
    print("Preparing for inductive test...")

    # load data after preprocess
    loaddata = open("{}/data_tKG.pkl".format(dataset), "rb")
    data = pickle.load(loaddata)
    loaddata.close()

    loadsr2o = open("{}/sr2o_all_tKG.pkl".format(dataset), "rb")
    sr2o = pickle.load(loadsr2o)
    loadsr2o.close()

    loadso2r = open("{}/so2r_all_tKG.pkl".format(dataset), "rb")
    so2r = pickle.load(loadso2r)
    loadso2r.close()

    loadtriples = open("{}/triples_tKG.pkl".format(dataset), "rb")
    triples = pickle.load(loadtriples)
    loadtriples.close()

    loadadjs = open("{}/adjs_tKG.pkl".format(dataset), "rb")
    adjs = pickle.load(loadadjs)
    loadadjs.close()

    loadtimestamp = open("{}/timestamp_tKG.pkl".format(dataset), "rb")
    timestamp = pickle.load(loadtimestamp)
    loadtimestamp.close()

    loadindep = open("{}/t_indep_trp.pkl".format(dataset), "rb")
    t_indep_trp = pickle.load(loadindep)
    loadindep.close()

    loadnei = open("{}/neighbor_tKG.pkl".format(dataset), "rb")
    neighbor = pickle.load(loadnei)
    loadnei.close()

    loadinduct = open("{}/inductive.pkl".format(dataset), "rb")
    induct = pickle.load(loadinduct)
    loadinduct.close()

    def get_total_number(inPath, fileName):
        with open(os.path.join(inPath, fileName), "r") as fr:
            for line in fr:
                line_split = line.split()
                return int(line_split[0]), int(line_split[1])

    num_e, num_rel = get_total_number("{}/".format(dataset), "stat.txt")
    logger.info("number of entities:" + str(num_e))
    logger.info("number of relations:" + str(num_rel))

    # timestamps
    # normalize timestamps, and scale
    ts_max = max(
        max(max(timestamp["train"]), max(timestamp["test"])), max(timestamp["valid"])
    )  # max timestamp in the dataset
    test_timestamps = (
        torch.tensor(timestamp["test"]) / torch.tensor(ts_max, dtype=torch.float)
    ) * scale

    print("number of testing snapshots:", len(timestamp["test"]))
    logger.info("number of testing snapshots:" + str(len(timestamp["test"])))

    # adjs
    test_adj = adjs["test"]

    # test triples
    test_triple = triples["test"]

    # one hop neighbor
    test_1nei = neighbor["test"]

    # so2r
    test_so2r = so2r["test"]

    # inductive evaluations
    induct_tar = induct

    return (
        num_e,
        num_rel,
        test_timestamps,
        test_adj,
        test_triple,
        test_1nei,
        t_indep_trp,
        test_so2r,
        induct_tar,
    )


def load_adjmtx(dataset):
    loadadjlist = open("{}/adjlist_tKG.pkl".format(dataset), "rb")
    adjlist = pickle.load(loadadjlist)
    loadadjlist.close()
    return adjlist
