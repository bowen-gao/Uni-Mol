# Copyright (c) DP Technology.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from xmlrpc.client import Boolean
import torch
from collections import OrderedDict
import unicore
import numpy as np
import faiss 
import lmdb
import pickle
import torch.nn.functional as F


from unicore.data import (
    Dictionary,
    NestedDictionaryDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenizeDataset,
    RightPadDataset2D,
    RawLabelDataset,
    RawArrayDataset,
    FromNumpyDataset,
)
from unimol.data import (
    KeyDataset,
    ConformerSampleDataset,
    DistanceDataset,
    EdgeTypeDataset,
    RemoveHydrogenDataset,
    AtomTypeDataset,
    NormalizeDataset,
    CroppingDataset,
    RightPadDatasetCoord,
    data_utils,
)

from unimol.data.tta_dataset import TTADataset
from unicore.tasks import UnicoreTask, register_task


logger = logging.getLogger(__name__)

task_metainfo = {
    "esol": {
        "mean": -3.0501019503546094,
        "std": 2.096441210089345,
        "target_name": "logSolubility",
    },
    "freesolv": {
        "mean": -3.8030062305295944,
        "std": 3.8478201171088138,
        "target_name": "freesolv",
    },
    "lipo": {"mean": 2.186336, "std": 1.203004, "target_name": "lipo"},
    "qm7dft": {
        "mean": -1544.8360893118609,
        "std": 222.8902092792289,
        "target_name": "u0_atom",
    },
    "qm8dft": {
        "mean": [
            0.22008500524052105,
            0.24892658759891675,
            0.02289283121913152,
            0.043164444107224746,
            0.21669716560818883,
            0.24225989336408812,
            0.020287111373358993,
            0.03312609817084387,
            0.21681478862847584,
            0.24463634931699113,
            0.02345177178004201,
            0.03730141834205415,
        ],
        "std": [
            0.043832862248693226,
            0.03452326954549232,
            0.053401140662012285,
            0.0730556474716259,
            0.04788020599385645,
            0.040309670766319,
            0.05117163534626215,
            0.06030064428723054,
            0.04458294838213221,
            0.03597696243350195,
            0.05786865052149905,
            0.06692733477994665,
        ],
        "target_name": [
            "E1-CC2",
            "E2-CC2",
            "f1-CC2",
            "f2-CC2",
            "E1-PBE0",
            "E2-PBE0",
            "f1-PBE0",
            "f2-PBE0",
            "E1-CAM",
            "E2-CAM",
            "f1-CAM",
            "f2-CAM",
        ],
    },
    "qm9dft": {
        "mean": [-0.23997669940621352, 0.011123767412331285, 0.2511003712141015],
        "std": [0.02213143402267657, 0.046936069870866196, 0.04751888787058615],
        "target_name": ["homo", "lumo", "gap"],
    },
}

def similarity_search(x, y, dim, normalize=False):
    x = np.float32(x)
    y = np.float32(y)
    num = x.shape[0]
    idx = faiss.IndexFlatL2(dim)
    if normalize:
        faiss.normalize_L2(x)
        faiss.normalize_L2(y)
    idx.add(x)
    scores, prediction = idx.search(y, 10)
    return scores, prediction



@register_task("mol_finetune_active")
class UniMolFinetuneActiveTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", help="downstream data path")
        parser.add_argument("--task-name", type=str, help="downstream task name")
        parser.add_argument(
            "--classification-head-name",
            default="classification",
            help="finetune downstream task name",
        )
        parser.add_argument(
            "--num-classes",
            default=1,
            type=int,
            help="finetune downstream task classes numbers",
        )
        parser.add_argument("--reg", action="store_true", help="regression task")
        parser.add_argument("--no-shuffle", action="store_true", help="shuffle data")
        parser.add_argument(
            "--conf-size",
            default=10,
            type=int,
            help="number of conformers generated with each molecule",
        )
        parser.add_argument(
            "--remove-hydrogen",
            action="store_true",
            help="remove hydrogen atoms",
        )
        parser.add_argument(
            "--remove-polar-hydrogen",
            action="store_true",
            help="remove polar hydrogen atoms",
        )
        parser.add_argument(
            "--max-atoms",
            type=int,
            default=256,
            help="selected maximum number of atoms in a molecule",
        )
        parser.add_argument(
            "--dict-name",
            default="dict.txt",
            help="dictionary file",
        )
        parser.add_argument(
            "--only-polar",
            default=1,
            type=int,
            help="1: only reserve polar hydrogen; 0: no hydrogen; -1: all hydrogen ",
        )
        parser.add_argument(
            "--seed-data-ratio",
            default=0.02,
            type=float,
            help="seed data ratio",
        )
        parser.add_argument(
            "--ft-steps",
            default=100,
            type=int,
            help="finetuning steps",
        )
        parser.add_argument(
            "--aug-steps",
            default=4,
            type=int,
            help="data augmentation steps",
        )
        parser.add_argument(
            "--label-budget-ratio",
            default=0.2,
            type=float,
            help="max ratio of augmented data",
        )
        parser.add_argument(
            "--adv-lambda",
            default=0.5,
            type=float,
            help="step size for adv attack",
        )
        parser.add_argument(
            "--random-gaussian",
            default=False,
            type=Boolean,
            help="whether use gaussian noise instead of adv attack",
        )
        parser.add_argument(
            "--random-active-learning",
            default=False,
            type=Boolean,
            help="whether use random retrieval for active learning",
        )
        parser.add_argument(
            "--uncertainty-sampling",
            default=False,
            type=Boolean,
            help="whether use uncertainty sampling for active learning",
        )
    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        if self.args.only_polar > 0:
            self.args.remove_polar_hydrogen = True
        elif self.args.only_polar < 0:
            self.args.remove_polar_hydrogen = False
        else:
            self.args.remove_hydrogen = True
        if self.args.task_name in task_metainfo:
            # for regression task, pre-compute mean and std
            self.mean = task_metainfo[self.args.task_name]["mean"]
            self.std = task_metainfo[self.args.task_name]["std"]
        self.tstep = 0
        self.candidate_features = None
        self.all_features = None
        self.batch_size = args.batch_size
        self.fdata, self.len_data = self.load_local_dataset("train")
        self.fdata = next(iter(self.fdata))
        self.fdata = unicore.utils.move_to_cuda(self.fdata)
        self.full_net = self.fdata["net_input"]
        self.full_target = self.fdata["target"]
        self.full_smi = self.fdata["smi_name"]
        
        self.seed_net, self.seed_target, self.seed_smi, selected = self.load_seed_data()
        self.seed_smi = list(self.seed_smi)
        self.scount = len(selected)
        self.scount_init = len(selected)
        self.used_id_list = selected.tolist()
        
        
        self.sdata_ft_sampled = torch.ones(self.len_data).int().cuda()
        self.sdata_aug_sampled = torch.ones(self.len_data).int().cuda()
        
        self.label_budgt = int(self.args.label_budget_ratio * self.len_data)

        self.mode = "ft"
        self.cur_aug_steps = 0
        self.cur_ft_steps = 0

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def load_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        if split == "train":
            split_path = os.path.join(self.args.data, self.args.task_name, split + "_seed.lmdb")
        else:
            split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        print("6666", type(dataset))
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        if not self.args.no_shuffle and split == "train":
            with data_utils.numpy_seed(self.args.seed):
                shuffle = np.random.permutation(len(src_dataset))

            self.datasets[split] = SortDataset(
                nest_dataset,
                sort_order=[shuffle],
            )
        else:
            self.datasets[split] = nest_dataset
    
    def load_local_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        dataset = LMDBDataset(split_path)
        if split == "train":
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")
            sample_dataset = ConformerSampleDataset(
                dataset, self.args.seed, "atoms", "coordinates"
            )
            dataset = AtomTypeDataset(dataset, sample_dataset)
        else:
            dataset = TTADataset(
                dataset, self.args.seed, "atoms", "coordinates", self.args.conf_size
            )
            dataset = AtomTypeDataset(dataset, dataset)
            tgt_dataset = KeyDataset(dataset, "target")
            smi_dataset = KeyDataset(dataset, "smi")

        dataset = RemoveHydrogenDataset(
            dataset,
            "atoms",
            "coordinates",
            self.args.remove_hydrogen,
            self.args.remove_polar_hydrogen,
        )
        dataset = CroppingDataset(
            dataset, self.seed, "atoms", "coordinates", self.args.max_atoms
        )
        dataset = NormalizeDataset(dataset, "coordinates", normalize_coord=True)
        src_dataset = KeyDataset(dataset, "atoms")
        src_dataset = TokenizeDataset(
            src_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        coord_dataset = KeyDataset(dataset, "coordinates")

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        edge_type = EdgeTypeDataset(src_dataset, len(self.dictionary))
        coord_dataset = FromNumpyDataset(coord_dataset)
        coord_dataset = PrependAndAppend(coord_dataset, 0.0, 0.0)
        distance_dataset = DistanceDataset(coord_dataset)

        nest_dataset = NestedDictionaryDataset(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        src_dataset,
                        pad_idx=self.dictionary.pad(),
                    ),
                    "src_coord": RightPadDatasetCoord(
                        coord_dataset,
                        pad_idx=0,
                    ),
                    "src_distance": RightPadDataset2D(
                        distance_dataset,
                        pad_idx=0,
                    ),
                    "src_edge_type": RightPadDataset2D(
                        edge_type,
                        pad_idx=0,
                    ),
                },
                "target": {
                    "finetune_target": RawLabelDataset(tgt_dataset),
                },
                "smi_name": RawArrayDataset(smi_dataset),
            },
        )
        len_data = len(nest_dataset)
        return torch.utils.data.DataLoader(nest_dataset, batch_size=len_data, collate_fn=nest_dataset.collater), len_data
    
    def encode_features(self, model):
        import time
        start_time = time.time()
        
        
        bsz = 16
        block = self.len_data // bsz 
        candidate_features = np.zeros((self.len_data, 512), float)
        for i in range(block+1):
            if i < block:
                l = i*bsz
                r = (i+1)*bsz
            else:
                l = i*bsz
                r = self.len_data
            net_input = OrderedDict()
            for k,v in self.full_net.items():
                net_input[k] = v[l:r]

            encode_rep, _ = model(
                **net_input,
                features_only=True,
                classification_head_name=self.args.classification_head_name,
                encode_mode=True
            )

            feature = encode_rep.mean(dim=1)
            candidate_features[l:r] = feature.detach().float().cpu().numpy()

        # print("candidate_features", candidate_features.shape)
        print("--- %s seconds ---" % (time.time() - start_time))
        return candidate_features
    
    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
    
    
    def set_sample_used(self, used_id, defult_used_value = 0):
        self.candidate_features[used_id] = defult_used_value
        
        used_id = int(used_id)
        if used_id not in self.used_id_list:
            self.used_id_list.append(used_id)
        return 0


    
    def load_seed_data(self):
        count = int(self.args.seed_data_ratio * self.len_data) 
        cand = torch.ones(self.len_data).long().cuda()
        prob = cand / cand.sum(dim=-1)
        selected = prob.multinomial(num_samples=count, replacement=False)
        # print(selected)
        
        seed_net = OrderedDict()
        for k,v in self.full_net.items():
            seed_net[k] = torch.zeros_like(v)
            copy_data = v.index_select(dim=0, index=selected)
            seed_net[k][:count] = copy_data
        
        seed_target = OrderedDict()
        for k,v in self.full_target.items():
            seed_target[k] = torch.zeros_like(v)
            copy_data = v.index_select(dim=0, index=selected)
            seed_target[k][:count] = copy_data 
 
        seed_smi = np.array(self.full_smi)[selected.cpu().detach().numpy()]
        

        return seed_net, seed_target, seed_smi, selected

    
    def retrieve_data_sample(self, mode='ft'):
        bsz = self.batch_size
        
        if mode == 'ft':
            cand = self.sdata_ft_sampled[:self.scount]

            if cand.sum(dim=-1) < bsz:
                self.sdata_ft_sampled = torch.ones_like(self.sdata_ft_sampled)
                cand = self.sdata_ft_sampled[:self.scount]
        else:
            cand = self.sdata_aug_sampled[:self.scount]

            if cand.sum(dim=-1) < bsz:
                self.sdata_aug_sampled = torch.ones_like(self.sdata_aug_sampled)
                cand = self.sdata_aug_sampled[:self.scount]
        
        

        prob = cand / cand.sum(dim=-1)
        selected = prob.multinomial(num_samples=bsz, replacement=False)
        
        # print(selected)


        if mode == 'ft':
            self.sdata_ft_sampled[selected] = 0 
        else:
            self.sdata_aug_sampled[selected] = 0

        net_input = OrderedDict()

        target = OrderedDict()

        for k,v in self.seed_net.items():
            net_input[k] = v.index_select(dim=0, index=selected)
        for k,v in self.seed_target.items():
            target[k] = v.index_select(dim=0, index=selected)

        sample = {
            "net_input" : net_input,
            "target" : target
        }
        return sample
    
    
    def create_adversarial_sample(self, model):
        sample = self.retrieve_data_sample(mode='aug')

        feature, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            encode_mode=True
        )
        #feature = torch.squeeze(encode_rep[:, 0, :], 1)
        # print("encode_out", feature.shape)
        classifier = model.classification_heads[self.args.classification_head_name]
        if self.args.random_gaussian:
            noise = torch.normal(mean=0, std=1, size=feature.size()).type_as(feature)
            update_feature = feature + 0.1 * noise
            update_feature = feature.mean(dim=1)
        else:
            update_feature = self.adversarial_attack(model, classifier, sample, feature)
        
        update_feature = update_feature.detach().float().cpu().numpy()
        #print(update_feature)
        scores, pred = similarity_search(self.candidate_features, update_feature, feature.size(-1), normalize=True)

        if self.args.random_active_learning:
            cand = torch.ones(self.len_data).long().cuda()
            for idx in self.used_id_list:
                cand[idx] = 0
            prob = cand / cand.sum(dim=-1)
            selected = prob.multinomial(num_samples=self.batch_size, replacement=False)
            for adv_idx in selected:
                for k,v in self.full_net.items():
                    self.seed_net[k][self.scount] = v[adv_idx]

                for k,v in self.full_target.items():
                    self.seed_target[k][self.scount] = v[adv_idx]
                self.set_sample_used(adv_idx)
                self.scount += 1  
        else:
            for idx in range(update_feature.shape[0]):
                found = 0
                for i in range(len(pred[idx])):
                    adv_idx = pred[idx][i]
                    if adv_idx not in self.used_id_list:
                        found = 1
                        break
                if found:
                    for k,v in self.full_net.items():
                        self.seed_net[k][self.scount] = v[adv_idx]

                    for k,v in self.full_target.items():
                        self.seed_target[k][self.scount] = v[adv_idx]
                    self.set_sample_used(adv_idx)
                    self.scount += 1        
        
        return self.scount 

    def create_uncertainty_sampling_sample(self, model):

        cans = []
        for idx in range(self.len_data):
            if idx not in self.used_id_list:
                cans.append(idx)
        
        bsz = 16
        block = len(cans) // bsz 
        entropy = np.zeros(len(cans), float)
        for i in range(block+1):
            if i < block:
                l = i*bsz
                r = (i+1)*bsz
            else:
                l = i*bsz
                r = len(cans)
            net_input = OrderedDict()
            target = OrderedDict()
            for k,v in self.full_net.items():
                net_input[k] = v[cans[l:r]]
            for k,v in self.full_target.items():
                target[k] = v[cans[l:r]]

            logits, _, _, _, _ = model(
                **net_input,
                classification_head_name=self.args.classification_head_name,
            )
            not_valid = target["finetune_target"] <= -0.5
            pred = torch.sigmoid(logits)
            entropy_cur = - (pred * pred.log() + (1-pred) * (1-pred).log())
            entropy_cur[not_valid] = 0
            entropy_cur = entropy_cur.mean(dim=-1)
            if i < block:
                entropy[i*bsz:(i+1)*bsz] = entropy_cur.detach().float().cpu().numpy()
            else:
                entropy[i*bsz:] = entropy_cur.detach().float().cpu().numpy()


       
        # selected = entropy.topk(k=self.batch_size, largest=True)[1]
        selected = np.argsort(entropy)[-self.batch_size:]
        print(selected, entropy[selected])
        #print()


        for idx in range(self.batch_size):
            adv_idx = cans[selected[idx]]
            for k,v in self.full_net.items():
                self.seed_net[k][self.scount] = v[adv_idx]

            for k,v in self.full_target.items():
                self.seed_target[k][self.scount] = v[adv_idx]
            self.set_sample_used(adv_idx)
            self.scount += 1        
        return self.scount 
    
    def adversarial_attack(self, model, classifier, sample, feature):
        feature = feature.detach()
        feature.requires_grad = True 

        model.train()
        logit_output = classifier(feature)
        is_valid = sample["target"]["finetune_target"] > -0.5
        pred = logit_output[is_valid].float()
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum",
        )

        model.zero_grad()
        loss.backward()
        avg_fnorm = feature.norm(dim=-1).mean() / feature.grad.norm(dim=-1).mean()
        update_feature = feature + self.args.adv_lambda * avg_fnorm  * feature.grad 
        update_feature = update_feature.mean(dim=1)
        return update_feature 
    
    def train_step(
        self, sample, model, loss, optimizer, update_num, ignore_grad=False
    ):
        """
        Do forward and backward, and return the loss as computed by *loss*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~unicore.data.UnicoreDataset`.
            model (~unicore.models.BaseUnicoreModel): the model
            loss (~unicore.losses.UnicoreLoss): the loss
            optimizer (~unicore.optim.UnicoreOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        print(self.tstep, self.scount)
        while self.tstep % (self.args.aug_steps + self.args.ft_steps) > self.args.ft_steps:
            if self.candidate_features is None:
                self.candidate_features = self.encode_features(model)
                self.all_features = self.candidate_features
                for idx in self.used_id_list:
                    self.set_sample_used(idx)
            
            if self.scount - self.scount_init < self.label_budgt:
                if self.args.uncertainty_sampling:
                    self.create_uncertainty_sampling_sample(model)
                else:
                    self.create_adversarial_sample(model)
                
            self.tstep += 1 
        self.candidate_features = None
        sample = self.retrieve_data_sample(mode='ft')
        self.tstep += 1
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
        

    
