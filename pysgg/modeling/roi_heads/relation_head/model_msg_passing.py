# modified from https://github.com/rowanz/neural-motifs
from torch import (
    cat as torch_cat,
    zeros as torch_zeros,
    arange as torch_arange,
    int64 as torch_int64,
    no_grad as torch_no_grad,
    equal as torch_equal,
)
from torch.nn.functional import relu as F_relu, softmax as F_softmax
from torch.nn import Module, Sequential, ReLU, BatchNorm1d, TransformerEncoderLayer, TransformerEncoder, GRUCell, Sigmoid, Embedding

from pysgg.data import get_dataset_statistics
from pysgg.modeling.make_layers import make_fc
from pysgg.modeling.roi_heads.relation_head.utils_relation import get_box_pair_info, get_box_info_norm, \
    layer_init
# from pysgg.modeling.utils import torch_cat
from .utils_motifs import obj_edge_vectors, encode_box_info


METHODS_DATA_2D = {'concat', 'raw_obj_pairwise'}
METHODS_DATA_1D = {'hadamard', 'mm', 'cosine_similarity'}


class IMPContext(Module):
    def __init__(self, config, in_channels, hidden_dim=512, num_iter=3):
        super(IMPContext, self).__init__()
        self.cfg = config

        self.pooling_dim = config.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim = hidden_dim
        self.num_iter = num_iter

        self.pairwise_feature_extractor = PairwiseFeatureExtractor(config,
                                                                   in_channels)

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        self.obj_unary = make_fc(self.pooling_dim, hidden_dim)
        self.edge_unary = make_fc(self.pooling_dim, hidden_dim)

        self.edge_gru = GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)
        self.node_gru = GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        self.sub_vert_w_fc = Sequential(make_fc(hidden_dim * 2, 1), Sigmoid())
        self.obj_vert_w_fc = Sequential(make_fc(hidden_dim * 2, 1), Sigmoid())
        self.out_edge_w_fc = Sequential(make_fc(hidden_dim * 2, 1), Sigmoid())
        self.in_edge_w_fc = Sequential(make_fc(hidden_dim * 2, 1), Sigmoid())

    def forward(self, inst_features, proposals, union_features, rel_pair_idxs, logger=None):
        num_objs = [len(b) for b in proposals]

        augment_obj_feat, rel_feats = self.pairwise_feature_extractor(inst_features, union_features,
                                                                      proposals, rel_pair_idxs, )

        obj_rep = self.obj_unary(augment_obj_feat)
        rel_rep = F_relu(self.edge_unary(rel_feats))

        obj_count = obj_rep.shape[0]
        rel_count = rel_rep.shape[0]

        # generate sub-rel-obj mapping
        sub2rel = torch_zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj2rel = torch_zeros(obj_count, rel_count).to(obj_rep.device).float()
        obj_offset = 0
        rel_offset = 0
        sub_global_inds = []
        obj_global_inds = []
        for pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            num_rel = pair_idx.shape[0]
            sub_idx = pair_idx[:, 0].contiguous().long().view(-1) + obj_offset
            obj_idx = pair_idx[:, 1].contiguous().long().view(-1) + obj_offset
            rel_idx = torch_arange(num_rel, device=obj_rep.device, dtype=torch_int64).view(-1) + rel_offset

            sub_global_inds.append(sub_idx)
            obj_global_inds.append(obj_idx)

            sub2rel[sub_idx, rel_idx] = 1.0
            obj2rel[obj_idx, rel_idx] = 1.0

            obj_offset += num_obj
            rel_offset += num_rel

        sub_global_inds = torch_cat(sub_global_inds, dim=0)
        obj_global_inds = torch_cat(obj_global_inds, dim=0)

        # iterative message passing
        hx_obj = torch_zeros(obj_count, self.hidden_dim, requires_grad=False, device=obj_rep.device).float()
        hx_rel = torch_zeros(rel_count, self.hidden_dim, requires_grad=False, device=obj_rep.device).float()

        vert_factor = [self.node_gru(obj_rep, hx_obj)]
        edge_factor = [self.edge_gru(rel_rep, hx_rel)]

        for i in range(self.num_iter):
            # compute edge context
            sub_vert = vert_factor[i][sub_global_inds]
            obj_vert = vert_factor[i][obj_global_inds]
            weighted_sub = self.sub_vert_w_fc(
                torch_cat((sub_vert, edge_factor[i]), 1)) * sub_vert
            weighted_obj = self.obj_vert_w_fc(
                torch_cat((obj_vert, edge_factor[i]), 1)) * obj_vert

            edge_factor.append(self.edge_gru(weighted_sub + weighted_obj, edge_factor[i]))

            # Compute vertex context
            pre_out = self.out_edge_w_fc(torch_cat((sub_vert, edge_factor[i]), 1)) * edge_factor[i]
            pre_in = self.in_edge_w_fc(torch_cat((obj_vert, edge_factor[i]), 1)) * edge_factor[i]
            vert_ctx = sub2rel @ pre_out + obj2rel @ pre_in
            vert_factor.append(self.node_gru(vert_ctx, vert_factor[i]))

        return vert_factor[-1], edge_factor[-1]


class PairwiseFeatureExtractor(Module):
    """
    extract the pairwise features from the object pairs and union features.
    most pipeline keep same with the motifs instead the lstm massage passing process
    """

    def __init__(self, config, in_channels):
        super(PairwiseFeatureExtractor, self).__init__()
        self.cfg = config
        self.pairwise_detach = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.DETACH
        self.using_explicit_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.USING_EXPLICIT_PAIRWISE
        self.explicit_pairwise_data = pairwise_method_data = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_DATA
        self.explicit_pairwise_func = explicit_pairwise_func = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.EXPLICIT_PAIRWISE_FUNC

        statistics = get_dataset_statistics(config)
        obj_classes, rel_classes = statistics['obj_classes'], statistics['rel_classes']
        self.num_obj_classes = num_obj_classes = len(obj_classes)
        self.num_rel_classes = len(rel_classes)
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes

        # mode
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                self.mode = 'predcls'
            else:
                self.mode = 'sgcls'
        else:
            self.mode = 'sgdet'

        # features augmentation for instance features
        # word embedding
        # add language prior representation according to the prediction distribution
        # of objects
        self.embed_dim = embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.obj_dim = in_channels
        self.hidden_dim = hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim = pooling_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        # Pairwise
        if explicit_pairwise_func == 'mha':
            assert pairwise_method_data in METHODS_DATA_1D
            num_head_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_HEAD
            num_layers_pairwise = config.MODEL.ROI_RELATION_HEAD.PAIRWISE.MHA.NUM_LAYERS
            self.explicit_pairwise_func = Sequential(
                TransformerEncoder(TransformerEncoderLayer(d_model=hidden_dim, nhead=num_head_pairwise), num_layers_pairwise),
                ReLU(inplace=True),
                make_fc(hidden_dim, pooling_dim),
                ReLU(inplace=True),
            )

        self.word_embed_feats_on = self.cfg.MODEL.ROI_RELATION_HEAD.WORD_EMBEDDING_FEATURES
        if self.word_embed_feats_on:
            obj_embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
            self.obj_embed_on_prob_dist = Embedding(num_obj_classes, embed_dim)
            self.obj_embed_on_pred_label = Embedding(num_obj_classes, embed_dim)
            with torch_no_grad():
                self.obj_embed_on_prob_dist.weight.copy_(obj_embed_vecs, non_blocking=True)
                self.obj_embed_on_pred_label.weight.copy_(obj_embed_vecs, non_blocking=True)
        else:
            self.embed_dim = 0

        # features augmentation for rel pairwise features
        self.rel_feature_type = config.MODEL.ROI_RELATION_HEAD.EDGE_FEATURES_REPRESENTATION

        # the input dimension is ROI head MLP, but the inner module is pooling dim, so we need
        # to decrease the dimension first.
        if self.pooling_dim != in_channels:
            self.rel_feat_dim_not_match = True
            self.rel_feature_up_dim = make_fc(in_channels, pooling_dim)
            layer_init(self.rel_feature_up_dim, xavier=True)
        else:
            self.rel_feat_dim_not_match = False

        self.pairwise_obj_feat_updim_fc = make_fc(hidden_dim + self.obj_dim + embed_dim,
                                                  hidden_dim * 2)

        self.outdim = pooling_dim
        # position embedding
        # encode the geometry information of bbox in relationships
        self.geometry_feat_dim = 128
        self.pos_embed = Sequential(*[
            make_fc(9, 32), BatchNorm1d(32, momentum=0.001),
            make_fc(32, self.geometry_feat_dim), ReLU(inplace=True),
        ])

        if self.rel_feature_type in {"obj_pair", "fusion"}:
            self.spatial_for_vision = config.MODEL.ROI_RELATION_HEAD.CAUSAL.SPATIAL_FOR_VISION
            if self.spatial_for_vision:
                self.spt_emb = Sequential(*[make_fc(32, self.hidden_dim),
                                               ReLU(inplace=True),
                                               make_fc(self.hidden_dim, self.hidden_dim * 2),
                                               ReLU(inplace=True)
                                               ])
                layer_init(self.spt_emb[0], xavier=True)
                layer_init(self.spt_emb[2], xavier=True)

            self.pairwise_rel_feat_finalize_fc = Sequential(
                make_fc(self.hidden_dim * 2, self.pooling_dim),
                ReLU(inplace=True),
            )

        # map bidirectional hidden states of dimension self.hidden_dim*2 to self.hidden_dim
        self.obj_hidden_linear = make_fc(self.obj_dim + self.embed_dim + self.geometry_feat_dim, self.hidden_dim)

        self.obj_feat_aug_finalize_fc = Sequential(
            make_fc(self.hidden_dim + self.obj_dim + self.embed_dim, self.pooling_dim),
            ReLU(inplace=True),
        )

        # untreated average features

    # def moving_average(self, holder, input):
    #     assert len(input.shape) == 2
    #     with torch_no_grad():
    #         holder = holder * (1 - self.average_ratio) + self.average_ratio * input.mean(0).view(-1)
    #     return holder

    def pairwise_rel_features(self, augment_obj_feat, rel_pair_idxs, inst_proposals):
        hidden_dim = self.hidden_dim
        spatial_for_vision = self.spatial_for_vision
        obj_boxs = [get_box_info_norm(p.bbox, proposal=p) for p in inst_proposals]
        num_objs = [len(p) for p in inst_proposals]
        # post decode
        # (num_objs, hidden_dim) -> (num_objs, hidden_dim * 2)
        # going to split single object representation to sub-object role of relationship
        pairwise_obj_feats_fused = self.pairwise_obj_feat_updim_fc(augment_obj_feat)
        del augment_obj_feat
        pairwise_obj_feats_fused = pairwise_obj_feats_fused.view(pairwise_obj_feats_fused.size(0), 2, hidden_dim)
        head_rep = pairwise_obj_feats_fused[:, 0].contiguous().view(-1, hidden_dim)
        tail_rep = pairwise_obj_feats_fused[:, 1].contiguous().view(-1, hidden_dim)
        del pairwise_obj_feats_fused, hidden_dim
        # split
        # explicit_pairwise_features = head_rep * tail_rep
        # head_reps = head_rep.split(num_objs, dim=0)
        # tail_reps = tail_rep.split(num_objs, dim=0)
        # explicit_pairwise_features = explicit_pairwise_features.split(num_objs, dim=0)
        # generate the pairwise object for relationship representation
        # (num_objs, hidden_dim) <rel pairing > (num_objs, hidden_dim)
        #   -> (num_rel, hidden_dim * 2)
        #   -> (num_rel, hidden_dim)
        # obj_pair_feat4rel_rep = []

        # if spatial_for_vision is True:
        #     pair_bboxs_info = []
        #     for pair_idx, obj_box in zip(rel_pair_idxs, obj_boxs):
        #         # obj_pair_feat4rel_rep.append(torch_cat((head_rep_each[pair_idx[:, 0]], tail_rep_each[pair_idx[:, 1]]), dim=-1))
        #         pair_bboxs_info.append(get_box_pair_info(obj_box[pair_idx[:, 0]], obj_box[pair_idx[:, 1]]))
        #     pair_bbox_geo_info = torch_cat(pair_bboxs_info, dim=0)
        # obj_pair_feat4rel_rep = torch_cat(obj_pair_feat4rel_rep, dim=0)  # (num_rel, hidden_dim * 2)


        rel_pair_idxs_global = []
        num_objs_culsum = 0
        # TODO: maybe use cumsum as an optimization?
        for rel_pair_idx, num_obj in zip(rel_pair_idxs, num_objs):
            rel_pair_idxs_global.append(rel_pair_idx + num_objs_culsum if num_objs_culsum > 0 else rel_pair_idx)
            num_objs_culsum += num_obj

        rel_pair_idxs_global = torch_cat(rel_pair_idxs_global)
        rel_pair_idxs_global_head = rel_pair_idxs_global[:, 0]
        rel_pair_idxs_global_tail = rel_pair_idxs_global[:, 1]
        del rel_pair_idxs_global

        head_rep = head_rep[rel_pair_idxs_global_head]
        tail_rep = tail_rep[rel_pair_idxs_global_tail]

        obj_pair_feat4rel_rep = torch_cat((head_rep, tail_rep), dim=-1)
        obj_boxs = torch_cat(obj_boxs, dim=0)
        pair_bbox_geo_info = get_box_pair_info(obj_boxs[rel_pair_idxs_global_head], obj_boxs[rel_pair_idxs_global_tail])
        del rel_pair_idxs_global_head, rel_pair_idxs_global_tail

        # assert torch_equal(obj_pair_feat4rel_rep, prod_rep)
        # assert torch_equal(pair_bbox_geo_info, pair_bboxs_info_new)
        # head_rep, tail_rep = prod_rep.hsplit(2)
        if self.using_explicit_pairwise is False:
            return obj_pair_feat4rel_rep, None

        if self.pairwise_detach is True:
            head_rep = head_rep.clone().detach()
            tail_rep = tail_rep.clone().detach()

        if self.explicit_pairwise_data == 'hadamard':
            pairwise_obj_ctx = head_rep * tail_rep
            del head_rep, tail_rep

        if spatial_for_vision is True:
            obj_pair_feat4rel_rep *= self.spt_emb(pair_bbox_geo_info)
            del pair_bbox_geo_info
        # breakpoint()
        # pairwise_obj_ctx = self.explicit_pairwise_func(pairwise_obj_ctx)
        # obj_pair_feat4rel_rep = self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)  # (num_rel, hidden_dim)

        return self.explicit_pairwise_func(pairwise_obj_ctx), self.pairwise_rel_feat_finalize_fc(obj_pair_feat4rel_rep)


    def forward(self, inst_roi_feats, union_features, inst_proposals, rel_pair_idxs, ):
        """

        :param inst_roi_feats: instance ROI features, list(Tensor)
        :param inst_proposals: instance proposals, list(BoxList())
        :param rel_pair_idxs:
        :return:
            obj_pred_logits obj_pred_labels 2nd time instance classification results
            obj_representation4rel, the objects features ready for the represent the relationship
        """
        # using label or logits do the label space embeddings
        if self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            obj_labels = torch_cat([proposal.get_field("labels") for proposal in inst_proposals], dim=0)
        else:
            obj_labels = None

        if self.word_embed_feats_on:
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
                obj_embed_by_pred_dist = self.obj_embed_on_prob_dist(obj_labels.long())
            else:
                obj_logits = torch_cat([proposal.get_field("predict_logits") for proposal in inst_proposals], dim=0).detach()
                obj_embed_by_pred_dist = F_softmax(obj_logits, dim=1) @ self.obj_embed_on_prob_dist.weight

        # box positive geometry embedding
        assert inst_proposals[0].mode == 'xyxy'
        pos_embed = self.pos_embed(encode_box_info(inst_proposals))

        # word embedding refine
        # batch_size = inst_roi_feats.shape[0]
        if self.word_embed_feats_on:
            obj_pre_rep = torch_cat((inst_roi_feats, obj_embed_by_pred_dist, pos_embed), -1)
        else:
            obj_pre_rep = torch_cat((inst_roi_feats, pos_embed), -1)
        # object level contextual feature
        augment_obj_feat = self.obj_hidden_linear(obj_pre_rep)  # map to hidden_dim

        # todo reclassify on the fused object features
        # Decode in order
        if self.mode != 'predcls':
            # todo: currently no redo classification on embedding representation,
            #       we just use the first stage object prediction
            obj_pred_labels = torch_cat([each_prop.get_field("pred_labels") for each_prop in inst_proposals], dim=0)
        else:
            assert obj_labels is not None
            obj_pred_labels = obj_labels

        # object labels space embedding from the prediction labels
        if self.word_embed_feats_on:
            obj_embed_by_pred_labels = self.obj_embed_on_pred_label(obj_pred_labels.long())

        # average action in test phrase for causal effect analysis
        if self.word_embed_feats_on:
            augment_obj_feat = torch_cat((obj_embed_by_pred_labels, inst_roi_feats, augment_obj_feat), -1)
        else:
            augment_obj_feat = torch_cat((inst_roi_feats, augment_obj_feat), -1)

        if self.rel_feature_type in {"obj_pair", "fusion"}:
            rel_features, explicit_pairwise_features = self.pairwise_rel_features(augment_obj_feat, rel_pair_idxs, inst_proposals)
            if self.rel_feature_type == "fusion":
                if self.rel_feat_dim_not_match:
                    union_features = self.rel_feature_up_dim(union_features)
                # breakpoint()
                rel_features = union_features + rel_features + explicit_pairwise_features

        elif self.rel_feature_type == "union":
            if self.rel_feat_dim_not_match:
                union_features = self.rel_feature_up_dim(union_features)
            rel_features = union_features

        else:
            assert False
        # mapping to hidden
        augment_obj_feat = self.obj_feat_aug_finalize_fc(augment_obj_feat)

        return augment_obj_feat, rel_features
