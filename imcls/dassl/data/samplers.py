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
    将一个batch划分成几段，每一小段是来自于同源域同标签的数据
    前8段，领域和标签都可以随机，后8段领域不同，标签相同
    在mixstyle里也就可以在每一个小段上随机了

    每一段里的领域先固定，毕竟原版mixstyle就是固定的，再做不固定的做消融研究（不固定域指的是后8段）

    '''
    def __init__(self, data_source, batch_size, n_ins):
        if batch_size < n_ins:        #batchsize必须比类别数大
            raise ValueError(
                "batch_size={} must be no less "
                "than n_ins={}".format(batch_size, n_ins)
            )

        self.data_source = data_source
        self.n_ins = n_ins  #将一个batch划分成几段，每一小段是来自于同源域同标签的数据

        # Keep track of image indices for each label           domain
        self.idx_label_dict = defaultdict(list)
        self.domain_dict = defaultdict(list)
        self.idx_domain = defaultdict(list)   #索引对应域标签
        self.idx_domain_label = defaultdict(lambda: defaultdict(list))
        self.label_dict = defaultdict(list)
        for i, item in enumerate(data_source):  # 对读入的数据地址按照领域进行划分，生成字典
            self.idx_label_dict[i].append(item.label)
            self.domain_dict[item.domain].append(i)     #
            self.idx_domain[i].append(item.domain)    # 按照索引存入域标签，这里就真的像是个字典，去查每个索引图片的领域标签。
            self.idx_domain_label[item.domain][item.label].append(i)
            self.label_dict[item.label].append(i)
        self.labels = list(self.label_dict.keys())
        self.domains = list(self.domain_dict.keys())

        # # Make sure each domain has equal number of images
        # if n_domain is None or n_domain <= 0:
        #     n_domain = len(self.domains)
        # assert batch_size % n_domain == 0
        self.n_img_per_segment = batch_size // self.n_ins  # 每个batch里  每个段要包含的数据个数8

        self.batch_size = batch_size
        # n_domain denotes number of domains sampled in a minibatch
        # self.n_domain = n_domain
        self.length = len(list(self.__iter__()))

    def __iter__(self):
        domain_dict = copy.deepcopy(self.domain_dict)
        idx_label_dict = copy.deepcopy(self.idx_label_dict)
        idx_domain_label = copy.deepcopy(self.idx_domain_label)
        idx_domain = copy.deepcopy(self.idx_domain)
        domains = defaultdict(list)
        labels = copy.deepcopy(self.labels)
        for label in labels:         # 7*3的
            for d in self.domains:
                domains[label].append(d)

        count = defaultdict(lambda: defaultdict())  # 计数
        for i in self.domains:
            for j in labels:
                count[i][j] = len(idx_domain_label[i][j])

        final_idxs = []
        stop_sampling = False

        repeat_use = defaultdict(lambda: defaultdict(list))   #用于第二段   当第二部分没有可用数据时，使用之前分配过的数据，这一部分不会用太多
        batch_order = 0                     # 计算到第几次batch了

        count_repeat = 0   #计算用到的重复数据
        while not stop_sampling:
            runing_id = 0
            selected_idxs = []   #一个batch里的选定idx
            for id_seg in range(self.n_ins):    #遍历段号（总共16段，前8段随机，后8段有域和标签的约束）

                seg_selected_idxs = []     #一段里的选定idx
                max_first_rechoose = 10    #前8小段，遇到不足时，重新选择标签最大次数，用来防止1个标签使得
                select_domain = 0
                select_label = 0

                if id_seg <self.n_ins//2:    #前8段
                    if not len(labels):
                        stop_sampling = True
                        runing_id =id_seg
                        break

                    select_label = random.sample(labels, 1)[0]
                    if not len(domains[select_label]):
                        choose_count = 0
                        while choose_count < max_first_rechoose:
                            select_label = random.sample(labels, 1)[0]
                            # print(select_label)
                            if len(domains[select_label]):
                                break
                            choose_count += 1

                        if choose_count == max_first_rechoose:
                            stop_sampling = True
                            runing_id = id_seg
                            break
                    select_domain = random.sample(domains[select_label], 1)[0]
                    idxs = idx_domain_label[select_domain][select_label]
                    seg_selected_idxs = random.sample(idxs, self.n_img_per_segment)  # 选择了一段的样本，8个
                    selected_idxs.extend(seg_selected_idxs)
                else:
                    old_idx = -self.n_ins // 2 * self.n_img_per_segment + 1
                    select_label = idx_label_dict[selected_idxs[old_idx]][0]
                    choose_domain = idx_domain[selected_idxs[old_idx]][0]  #前半段与之对应的部分的领域  前段已选中的领域
                    new_domains = copy.deepcopy(domains[select_label])
                    if choose_domain in new_domains:
                        new_domains.remove(choose_domain)

                    if not len(new_domains):  #剩余数据不足以构成了，利用之前已经分配的数据
                        new_domains = copy.deepcopy(self.domains)
                        new_domains.remove(choose_domain)
                        select_domain = random.sample(new_domains, 1)[0]  #选择新领域
                        idxs = []
                        for idx in repeat_use[select_domain][select_label]:
                            if idx < batch_order*self.batch_size:
                                idxs.extend(final_idxs[idx : idx+self.n_img_per_segment])
                        # for i in range(3):
                        #     for j in range(7):
                        #         print(repeat_use[i][j],'   ', end="")
                        #     print()

                        #print(select_domain,' ',select_label,' ',idxs, ' ',batch_order*self.batch_size)

                        seg_selected_idxs = random.sample(idxs, self.n_img_per_segment)
                        selected_idxs.extend(seg_selected_idxs)
                        if select_label in labels:
                            labels.remove(select_label)  #说明有两个域已经没有足够的数据了
                        count_repeat += self.n_img_per_segment
                    else:
                        select_domain = random.sample(new_domains, 1)[0]
                        idxs = idx_domain_label[select_domain][select_label]
                        seg_selected_idxs = random.sample(idxs, self.n_img_per_segment)  # 选择了一段的样本，8个
                        selected_idxs.extend(seg_selected_idxs)

                fin_idx = batch_order * self.batch_size + id_seg * self.n_img_per_segment  #final_idxs的每小段起始下标
                repeat_use[select_domain][select_label].append(fin_idx)



                for i in seg_selected_idxs:
                    if i in idx_domain_label[select_domain][select_label]:
                        idx_domain_label[select_domain][select_label].remove(i)
                count[select_domain][select_label] -= self.n_img_per_segment

                for la in self.labels:
                    for do in self.domains:
                        if count[do][la] < self.n_img_per_segment:
                            if do in domains[la]:
                                domains[la].remove(do)
                    # if len(domains[la]) == 1 and la in labels:
                    #     labels.remove(la)

            batch_order += 1
            if not stop_sampling:
                final_idxs.extend(selected_idxs)

            # if runing_id>0:
            #     del final_idxs[-(runing_id*self.n_img_per_segment):-1]
        ''' 验证
        for i in range(39):
            for j in range(64):
                k=i*128+j
                idx=final_idxs[k]
                idx1=final_idxs[k+64]
                if idx_label_dict[idx]!= idx_label_dict[idx1] or idx_domain[idx] == idx_domain[idx1]:
                    print('wrong')
        '''
        print()
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
        return Group_by_label(data_source, batch_size, n_ins)

    else:
        raise ValueError("Unknown sampler type: {}".format(sampler_type))
