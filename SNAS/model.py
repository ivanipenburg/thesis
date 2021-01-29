import torch
import torch.nn as nn
from operations import *
from torch.autograd import Variable
from utils import drop_path
import math
from genotypes import PRIMITIVES

from torch.utils import checkpoint as cp

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

        self.counter = 0

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        # 4
        self._steps = int((-3 + math.sqrt(9 + 8 * len(indices))) // 2)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        start = 0
        end = start + 2
        
        for i in range(self._steps):
            
            s = 0
            for j in range(start, end):
                h = states[self._indices[j]]
                op = self._ops[j]
                h = op(h)
                if self.training and drop_prob > 0.:
                    if not isinstance(op, Identity):
                        h = drop_path(h, drop_prob)
                s += h
            start = end
            end += (i + 3)
            states += [s]

        return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Module):
    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class NetworkCIFAR(nn.Module):
    def __init__(self, C, num_classes, layers, auxiliary, genotype, meta_arch):
        super(NetworkCIFAR, self).__init__()
        # Layers default is 20.
        self._layers = layers
        # True or False, if True, add auxiliary head.
        self._auxiliary = auxiliary

        self.test_counter = 0

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            # nn.Conv2d(in_channes, out_channels, kernel_size).
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        # nn.ModuleList holds submodules in a list.
        self.cells = nn.ModuleList()
        reduction_prev = False

        meta_arch = meta_arch.split("-")
        self.first_reduction = int(meta_arch[0])
        self.second_reduction = int(meta_arch[0]) + int(meta_arch[1]) + 1

        # Iterate over layers.
        for i in range(layers):
            # Check if layer is reduction layer.
            if i in [self.first_reduction, self.second_reduction]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False

            # Create cell.
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction

            # Add cell to nn.ModeluleList()
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

            # If second reduction layer.
            if i == self.second_reduction:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == self.second_reduction:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))

        return logits, logits_aux