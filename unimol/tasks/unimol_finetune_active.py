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
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

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
            default=0.3,
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
        parser.add_argument(
            "--fix-encoder",
            default=False,
            type=Boolean,
            help="whether fix encoder when finetuning the classifier",
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

        '''

        self.qm9data, self.qm9_len_data = self.load_qm9_dataset("train")
        self.qm9data = next(iter(self.qm9data))
        self.qm9data = unicore.utils.move_to_cuda(self.qm9data)
        self.qm9_net = self.qm9data["net_input"]
        for k,v in self.qm9_net.items():
            print(k, v.shape)
        for k,v in self.full_net.items():
            print(k, v.shape)
        self.qm9_target = self.qm9data["target"]
        print(self.qm9_target["finetune_target"].shape)
        self.qm9_smi = self.qm9data["smi_name"]
        '''
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
        self.used_aug_list = []
        self.pca = None
        self.pca_candidate_features = None

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
        # split_path = "/data/data/molecule/molecular_property_prediction/qm9/train.lmdb"
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
    
    def load_qm9_dataset(self, split, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the data scoure (e.g., train)
        """
        split_path = os.path.join(self.args.data, self.args.task_name, split + ".lmdb")
        split_path = "/home/gaobowen/666.lmdb"
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
            if i*bsz == self.len_data:
                break
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

            feature = encode_rep[:,0,:]
            candidate_features[l:r] = feature.detach().float().cpu().numpy()

        # print("candidate_features", candidate_features.shape)
        print("--- %s seconds ---" % (time.time() - start_time))

        self.pca = PCA(n_components=8)
        
        self.pca.fit(candidate_features)
        self.pca_candidate_features = self.pca.transform(candidate_features)


        return candidate_features
    
    def build_model(self, args):
        from unicore import models

        model = models.build_model(args, self)
        model.register_classification_head(
            self.args.classification_head_name,
            num_classes=self.args.num_classes,
        )
        return model
    
    
    def set_sample_used(self, used_id, defult_used_value = 1000):
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

    
    def retrieve_data_sample(self, mode='ft', bsz=None):
        if bsz is None:
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
    
    def retrieve_data_sample_aug(self, num_sample):
        
    
        cand = self.sdata_aug_sampled[:self.scount]

        if cand.sum(dim=-1) < num_sample:
            self.sdata_aug_sampled = torch.ones_like(self.sdata_aug_sampled)
            cand = self.sdata_aug_sampled[:self.scount]
        
        

        prob = cand / cand.sum(dim=-1)
        selected = prob.multinomial(num_samples=num_sample, replacement=False)
        
        # print(selected)



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
        sample = self.retrieve_data_sample(mode="aug", bsz = 3*self.batch_size)
        feature, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name=self.args.classification_head_name,
            encode_mode=True
        )

        classifier = model.classification_heads[self.args.classification_head_name]
        update_feature = self.adversarial_attack(model, classifier, sample, feature)
        update_feature = update_feature.detach().float().cpu().numpy()
        #scores, pred = similarity_search(self.candidate_features, update_feature, feature.size(-1), normalize=False)  
        pca_update_feature = self.pca.transform(update_feature)
        print(pca_update_feature.shape)
        scores, pred = similarity_search(self.pca_candidate_features, pca_update_feature, pca_update_feature.shape[-1], normalize=False)  
        '''
        cur_feature = feature[:,0,:].detach().float().cpu().numpy()
        original_dist = np.linalg.norm(self.candidate_features[pred[0][0]].reshape(1, -1) - cur_feature[0].reshape(1, -1))
        update_dist = np.linalg.norm(self.candidate_features[pred[0][0]].reshape(1, -1) - update_feature[0].reshape(1, -1))
        
        print("original_dist", original_dist)
        print("update_dist", update_dist)
        '''
        adv_cans = []
        for idx in range(update_feature.shape[0]):
            count = 0
            for i in range(len(pred[idx])):
                if count==1:
                    break
                adv_idx = pred[idx][i]
                score = scores[idx][i]
                if adv_idx not in self.used_id_list:
                    if adv_idx not in adv_cans:
                        adv_cans.append(adv_idx) 
                        count+=1
      
        # adv_cans = []
        adv_cans = np.array(adv_cans)
        
        randoms = []
        for idx in range(self.len_data):
            if idx not in self.used_id_list:
                if idx not in adv_cans:
                    randoms.append(idx)
        randoms = np.array(randoms)
        num_random = 4 * self.batch_size - len(adv_cans)
        # num_random = 4 * self.batch_size
        random_cans = np.random.choice(randoms, num_random, replace=False)
        cans = np.concatenate([adv_cans, random_cans])
        # cans = adv_cans
        print(len(adv_cans), len(random_cans))
        bsz = 16
        block = len(cans) // bsz 
        entropy = np.zeros(len(cans), float)
        for i in range(block+1):
            if i*bsz == len(cans):
                break
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
        adv_count = 0
        random_count = 0
        for idx in range(self.batch_size):
            adv_idx = cans[selected[idx]]
            if adv_idx in adv_cans:
                adv_count+=1
            else:
                random_count+=1
        print("ratio", adv_count, random_count)
        for idx in range(self.batch_size):
            adv_idx = cans[selected[idx]]
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
            if i*bsz == len(cans):
                break
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
        not_valid = sample["target"]["finetune_target"] <= -0.5
        pred = logit_output[is_valid].float()
        pred_sigmoid = torch.sigmoid(logit_output)
        entropy = - (pred_sigmoid * pred_sigmoid.log() + (1-pred_sigmoid) * (1-pred_sigmoid).log())
        entropy[not_valid] = 0
        entropy = entropy.mean(dim=-1, keepdim=True)
        targets = sample["target"]["finetune_target"][is_valid].float()
        loss = F.binary_cross_entropy_with_logits(
            pred,
            targets,
            reduction="sum",
        )

        model.zero_grad()
        loss.backward()
        feature_grad = feature.grad[:,0,:]
        feature = feature[:,0,:]
        avg_fnorm = feature.norm(dim=-1) / feature_grad.norm(dim=-1)
        avg_fnorm = avg_fnorm.view(-1, 1).expand(-1, feature_grad.shape[1])
        #print(entropy.shape, avg_fnorm.shape, feature.grad.shape)
        update_feature = feature + self.args.adv_lambda * avg_fnorm  * feature_grad
        # update_feature = update_feature.mean(dim=1)
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
        self.sdata_aug_sampled = torch.ones_like(self.sdata_aug_sampled)
        self.used_aug_list = []
        sample = self.retrieve_data_sample(mode='ft')
        self.tstep += 1
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = loss(model, sample, fix_encoder=self.args.fix_encoder)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)
        return loss, sample_size, logging_output
        

    
