import copy
import numpy as np
import random
from collections import defaultdict
from torch.utils.data.sampler import Sampler, RandomSampler, SequentialSampler


class RandomDomainSampler(Sampler):
    """Randomly samples N domains each with K images
    to form a minibatch of size N*K.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_domain (int): number of domains to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):    #对读入的数据地址按照领域进行划分，生成字典
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain   #每个batch里  每个域要包含的数据个数

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)  #随机选择几个域，这里默认是2个

            for domain in selected_domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:     #为了保证已经选择的数据，不会被二次选择，将他们移除
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:   #剩余的数据不够一次batch，抛弃
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class SeqDomainSampler(Sampler):
    """Sequential domain sampler, which randomly samples K
    images from each domain to form a minibatch.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
    """

    def __init__(self, data_source, batch_size):
        self.data_source = data_source

        # Keep track of image indices for each domain
        self.domain_dict = defaultdict(list)
        for i, item in enumerate(data_source):
            self.domain_dict[item.domain].append(i)
        self.domains = list(self.domain_dict.keys())
        self.domains.sort()

        # Make sure each domain has equal number of images
        n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            for domain in self.domains:
                idxs = domain_dict[domain]
                selected_idxs = random.sample(idxs, self.n_img_per_domain)
                final_idxs.extend(selected_idxs)

                for idx in selected_idxs:
                    domain_dict[domain].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:
                    stop_sampling = True

        return iter(final_idxs)

    def __len__(self):
        return self.length


class RandomClassSampler(Sampler):        #按照类别划分    分成了n_ins 16段，这是之后要用adarnn的那个，可以在这里改
    """Randomly samples N classes each with K instances to
    form a minibatch of size N*K.

    Modified from https://github.com/KaiyangZhou/deep-person-reid.

    Args:
        data_source (list): list of Datums.
        batch_size (int): batch size.
        n_ins (int): number of instances per class to sample in a minibatch.
    """

    def __init__(self, data_source, batch_size, n_ins):
        if batch_size < n_ins:
            raise ValueError(
                "batch_size={} must be no less "
                "than n_ins={}".format(batch_size, n_ins)
            )

        self.data_source = data_source
        self.batch_size = batch_size
        self.n_ins = n_ins
        self.ncls_per_batch = self.batch_size // self.n_ins
        self.index_dic = defaultdict(list)
        for index, item in enumerate(data_source):
            self.index_dic[item.label].append(index)
        self.labels = list(self.index_dic.keys())
        assert len(self.labels) >= self.ncls_per_batch

        # estimate number of images in an epoch
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for label in self.labels:
            idxs = copy.deepcopy(self.index_dic[label])
            if len(idxs) < self.n_ins:
                idxs = np.random.choice(idxs, size=self.n_ins, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.n_ins:
                    batch_idxs_dict[label].append(batch_idxs)
                    batch_idxs = []

        avai_labels = copy.deepcopy(self.labels)
        final_idxs = []

        while len(avai_labels) >= self.ncls_per_batch:
            selected_labels = random.sample(avai_labels, self.ncls_per_batch)
            for label in selected_labels:
                batch_idxs = batch_idxs_dict[label].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[label]) == 0:
                    avai_labels.remove(label)

        return iter(final_idxs)

    def __len__(self):
        return self.length

class Group_by_label(Sampler):        #修改的"RandomDomainSampler"    按照类别划分
    '''
        按照原先领域划分，这里也保持将一个batch，一分为2，每次随机选定两个域，在这两个域去对应相应的标签数据。
        第一段随机，第二段不随机，按照第一段对应的标签 领域标签去找第二段的内容
        记得在mixstyle里取消两部分分别的随机打乱
    '''
    def __init__(self, data_source, batch_size, n_domain):
        self.data_source = data_source

        # Keep track of image indices for each label           domain
        self.idx_label_dict = defaultdict(list)
        self.domain_dict = defaultdict(list)
        self.idx_domain = defaultdict(list)   #索引对应域标签
        self.idx_domain_label = defaultdict(lambda: defaultdict(list))
        # self.data_spilt_idx = np.full((3,7), 10000)    #建立了对不同领域和不同标签的起始值保存,有了idx_domain_label，这个可以不用了
        for i, item in enumerate(data_source):  # 对读入的数据地址按照领域进行划分，生成字典
            self.idx_label_dict[i].append(item.label)
            self.domain_dict[item.domain].append(i)     #
            self.idx_domain[i].append(item.domain)    # 按照索引存入域标签，这里就真的像是个字典，去查每个索引图片的领域标签。
            self.idx_domain_label[item.domain][item.label].append(i)
            # self.data_spilt_idx[item.domain][item.label]=min(self.data_spilt_idx[item.domain][item.label], i)
        # self.labels = list(self.label_dict.keys())

        self.domains = list(self.domain_dict.keys())

        # Make sure each domain has equal number of images
        if n_domain is None or n_domain <= 0:
            n_domain = len(self.domains)
        assert batch_size % n_domain == 0
        self.n_img_per_domain = batch_size // n_domain  # 每个batch里  每个域要包含的数据个数64

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        idx_label_dict = copy.deepcopy(self.idx_label_dict)
        idx_domain_label = copy.deepcopy(self.idx_domain_label)

        final_idxs = []
        stop_sampling = False

        while not stop_sampling:
            selected_domains = random.sample(self.domains, self.n_domain)  # 随机选择几个域，这里默认是2个

            flag = True    #第一组  先随机选择
            for domain in selected_domains:
                selected_idxs = []

                if flag:
                    idxs = domain_dict[domain]
                    selected_idxs = random.sample(idxs, self.n_img_per_domain)
                    first_select = selected_idxs
                else:
                    for x1 in first_select:      #第二组，后64  选择和第一组对应同标签但不同域的。其中域已经确定
                        x2 = random.sample(idx_domain_label[domain][idx_label_dict[x1][0]], 1)
                        selected_idxs.append(x2[0])
                        idx_domain_label[domain][idx_label_dict[x1][0]].remove(x2[0])
                    first_select = []

                final_idxs.extend(selected_idxs)


                for idx in selected_idxs:  # 为了保证已经选择的数据，不会被二次选择，将他们移除
                    # print(idx)
                    domain_dict[domain].remove(idx)
                    idx_label = idx_label_dict[idx][0]
                    if flag:
                        idx_domain_label[domain][idx_label].remove(idx)

                remaining = len(domain_dict[domain])
                if remaining < self.n_img_per_domain:  # 剩余的数据不够一次batch，抛弃
                    stop_sampling = True

                flag = False

        return iter(final_idxs)

    def __len__(self):
        return self.length



def build_sampler(
    sampler_type,
    cfg=None,
    data_source=None,
    batch_size=32,
    n_domain=0,
    n_ins=16
):
    if sampler_type == "RandomSampler":
        return RandomSampler(data_source)

    elif sampler_type == "SequentialSampler":
        return SequentialSampler(data_source)

    elif sampler_type == "RandomDomainSampler":
        return RandomDomainSampler(data_source, batch_size, n_domain)

    elif sampler_type == "SeqDomainSampler":
        return SeqDomainSampler(data_source, batch_size)

    elif sampler_type == "RandomClassSampler":
        return RandomClassSampler(data_source, batch_size, n_ins)

    elif sampler_type == "Group_by_label":
        return Group_by_label(data_source, batch_size, n_domain)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))
