import itertools

# import ipdb
import numpy as np
from torch.jit import script as torch_jit_script
from torch import (
    min as torch_min,
    max as torch_max,
    nonzero as torch_nonzero,
    equal as torch_equal,
    cat as torch_cat,
    clamp as torch_clamp,
    zeros as torch_zeros,
    tensor as torch_tensor,
    int64 as torch_int64,
)
from torch.nn.init import normal_, constant_, xavier_normal_, orthogonal_
from torch.nn.functional import softmax as F_softmax


@torch_jit_script
def get_box_info(boxes):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch_cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch_cat((boxes, center_box), 1)
    # if need_norm:
    #     breakpoint()
    #     box_info /= float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


def get_box_info_norm(boxes, proposal):
    """
    input: [batch_size, (x1,y1,x2,y2)]
    output: [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0
    center_box = torch_cat((boxes[:, :2] + 0.5 * wh, wh), 1)
    box_info = torch_cat((boxes, center_box), 1)
    box_info /= float(max(max(proposal.size[0], proposal.size[1]), 100))
    return box_info


@torch_jit_script
def get_box_pair_info(box1, box2):
    """
    input:
        box1 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
        box2 [batch_size, (x1,y1,x2,y2,cx,cy,w,h)]
    output:
        32-digits: [box1, box2, unionbox, intersectionbox]
    """
    # union box
    unionbox = box1[:, :4].clone()
    unionbox[:, [0, 1]] = torch_min(box1[:, [0, 1]], box2[:, [0, 1]])
    unionbox[:, [2,3]] = torch_max(box1[:, [2,3]], box2[:, [2,3]])

    union_info = get_box_info(unionbox)

    # intersection box
    intersection_box = box1[:, :4].clone()
    intersection_box[:, [0,1]] = torch_max(box1[:, [0,1]], box2[:, [0,1]])
    intersection_box[:, [2,3]] = torch_min(box1[:, [2,3]], box2[:, [2,3]])
    case1 = torch_nonzero(
        intersection_box[:, 2].contiguous().view(-1) < intersection_box[:, 0].contiguous().view(-1)).view(-1)
    case2 = torch_nonzero(
        intersection_box[:, 3].contiguous().view(-1) < intersection_box[:, 1].contiguous().view(-1)).view(-1)
    intersextion_info = get_box_info(intersection_box)
    if case1.numel() > 0:
        intersextion_info[case1, :] = 0
    if case2.numel() > 0:
        intersextion_info[case2, :] = 0
    return torch_cat((box1, box2, union_info, intersextion_info), 1)


@torch_jit_script
def nms_overlaps(boxes):
    """ get overlaps for each channel
    The overlapping of each box on each category
    return a tensor with N x N x C
    """
    assert boxes.dim() == 3
    N = boxes.size(0)
    nc = boxes.size(1)
    max_xy = torch_min(boxes[:, None, :, 2:].expand(N, N, nc, 2),
                       boxes[None, :, :, 2:].expand(N, N, nc, 2))

    min_xy = torch_max(boxes[:, None, :, :2].expand(N, N, nc, 2),
                       boxes[None, :, :, :2].expand(N, N, nc, 2))
    # the delta x,y in intersection of box pairs in same category
    inter = torch_clamp((max_xy - min_xy + 1.0), min=0)

    # n, n, 151
    inter = inter.prod(-1)
    boxes_flat = boxes.view(-1, 4)
    areas_flat = (boxes_flat[:, 2] - boxes_flat[:, 0] + 1.0) * (
            boxes_flat[:, 3] - boxes_flat[:, 1] + 1.0)
    areas = areas_flat.view(boxes.size(0), boxes.size(1))
    union = -inter + areas[None] + areas[:, None]
    return inter / union


def layer_init(layer, init_para=0.1, normal=False, xavier=True):
    xavier = False if normal == True else True
    if normal:
        normal_(layer.weight, mean=0, std=init_para)
        constant_(layer.bias, 0)
    elif xavier:
        xavier_normal_(layer.weight, gain=1.0)
        constant_(layer.bias, 0)


def obj_prediction_nms(boxes_per_cls, pred_logits, nms_thresh=0.3):
    """
    a global level non-maximum suppression,
    apply this on local level nms can get a better performance.

    boxes_per_cls:               [num_obj, num_cls, 4]
    pred_logits:                 [num_obj, num_category]
    """

    num_obj = pred_logits.shape[0]
    assert num_obj == boxes_per_cls.shape[0]
    # get the overlapping between all boxes pairs of each categories
    # (N, N, C) the box i and box j has overlapping on category c
    is_overlap = nms_overlaps(boxes_per_cls).view(boxes_per_cls.size(0),
                                                  boxes_per_cls.size(0),
                                                  boxes_per_cls.size(1)).cpu().numpy() >= nms_thresh

    prob_sampled = F_softmax(pred_logits, 1).detach().cpu().numpy()
    prob_sampled[:, 0] = 0  # set bg to 0

    pred_label = torch_zeros(num_obj, device=pred_logits.device, dtype=torch_int64)

    for i in range(num_obj):
        # take the global maximum score prediction boxes
        box_ind, cls_ind = np.unravel_index(prob_sampled.argmax(), prob_sampled.shape)
        if float(pred_label[int(box_ind)]) > 0:
            # if pred label bigger than 0 means it already assigned higher probability results
            pass
        else:
            pred_label[int(box_ind)] = int(cls_ind)
        # suppress all boxes overlapping and have same category with this maximum box
        prob_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
        prob_sampled[box_ind] = -1.0
        # Mark this box has already sampled so we won't re-sample
    return pred_label



def block_orthogonal(tensor, split_sizes, gain=1.0):
    sizes = list(tensor.size())
    if any([a % b != 0 for a, b in zip(sizes, split_sizes)]):
        raise ValueError("tensor dimensions must be divisible by their respective "
                         "split_sizes. Found size: {} and split_sizes: {}".format(sizes, split_sizes))
    indexes = [list(range(0, max_size, split))
               for max_size, split in zip(sizes, split_sizes)]
    # Iterate over all possible blocks within the tensor.
    for block_start_indices in itertools.product(*indexes):
        # A list of tuples containing the index to start at for this block
        # and the appropriate step size (i.e split_size[i] for dimension i).
        index_and_step_tuples = zip(block_start_indices, split_sizes)
        # This is a tuple of slices corresponding to:
        # tensor[index: index + step_size, ...]. This is
        # required because we could have an arbitrary number
        # of dimensions. The actual slices we need are the
        # start_index: start_index + step for each dimension in the tensor.
        block_slice = tuple([slice(start_index, start_index + step)
                             for start_index, step in index_and_step_tuples])

        # let's not initialize empty things to 0s because THAT SOUNDS REALLY BAD
        assert len(block_slice) == 2
        sizes = [x.stop - x.start for x in block_slice]
        tensor_copy = tensor.new(max(sizes), max(sizes))
        orthogonal_(tensor_copy, gain=gain)
        tensor[block_slice] = tensor_copy[0:sizes[0], 0:sizes[1]]


def percentile(t: torch_tensor, q: float):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """
    # Note that ``kthvalue()`` works one-based, i.e. the first sorted value
    # indeed corresponds to k=1, not k=0! Use float(q) instead of q directly,
    # so that ``round()`` returns an integer, even if q is a np.float32.
    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result
