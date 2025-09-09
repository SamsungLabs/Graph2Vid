import random
import torch
import torch.nn.functional as F
from torch import log, exp
import numpy as np
from networkx.algorithms.dag import lexicographical_topological_sort

from dp.exact_dp import drop_dtw
from dp.soft_dp import batch_dropDTW, batch_NW, batch_drop_dtw_machine, drop_dtw_machine, metadag2vid_soft
from dp.dp_utils import compute_all_costs
from dp.graph_utils import compute_metadag_costs, metadag2vid


def mil_nce(features_1=None, features_2=None, correspondance_mat=None, sim=None, eps=1e-5, gamma=1, hard_ratio=1):
    corresp = correspondance_mat.to(torch.float32)
    sim = features_1 @ features_2.T if sim is None else sim
    prod = sim / gamma
    # logsumexp trick happens here
    prod_exp = exp(prod - prod.max(dim=1, keepdim=True).values)
    nominator = (prod_exp * corresp).sum(dim=1)
    denominator = prod_exp.sum(dim=1)
    nll = -log(nominator / (denominator + eps) + eps)
    if hard_ratio < 1:
        n_hard_examples = int(nll.shape[0] * hard_ratio)
        hard_indices = nll.sort().indices[-n_hard_examples:]
        nll = nll[hard_indices]
    loss = nll.mean()
    if loss > 9999:
        import ipdb

        ipdb.set_trace()
    return loss


def compute_contrastive_loss(
    samples,
    l2_normalize=False,
    xz_gamma=10,
    object="node",
):
    # aggregating videos with attentino for their steps, i.e. done per each step
    all_frame_features, all_node_features = [], []
    frame_labels, node_labels = [], []
    seen_cls_ids = set()
    for i, sample in enumerate(samples):
        node_features, frame_features = sample[f"{object}_features"], sample["frame_features"]
        cls_id = sample["cls"]

        if l2_normalize:
            node_features = F.normalize(node_features, p=2, dim=1)
            frame_features = F.normalize(frame_features, p=2, dim=1)

        all_frame_features.append(frame_features)
        frame_labels.extend([cls_id] * sample["frame_features"].shape[0])

        if cls_id not in seen_cls_ids:
            all_node_features.append(node_features)
            node_labels.extend([cls_id] * sample[f"{object}_features"].shape[0])
            seen_cls_ids.add(cls_id)

    frame_labels, node_labels = torch.tensor(frame_labels), torch.tensor(node_labels)
    label_mat = (frame_labels[:, None] == node_labels[None, :]).to(frame_features.device)
    # reinforcing existing alignment with step descriptors

    all_frame_features = torch.cat(all_frame_features, dim=0)
    all_node_features = torch.cat(all_node_features, dim=0)
    zx_loss_row = mil_nce(all_frame_features, all_node_features, label_mat, gamma=xz_gamma)
    zx_loss_col = mil_nce(all_node_features, all_frame_features, label_mat.T, gamma=xz_gamma)
    zx_loss = (zx_loss_row + zx_loss_col) / 2
    return zx_loss


def compute_clust_loss(
    samples,
    distractors,
    l2_normalize=False,
    frame_gamma=10,
    xz_gamma=10,
    xz_hard_ratio=0.3,
    all_classes_distinct=False,
    bg_scope="global",
    object="step",
):
    # aggregating videos with attentino for their steps, i.e. done per each step
    all_pooled_frames, pooled_frames_labels = [], []
    all_step_features, all_step_labels = [], []
    global_step_id_count = 0
    for i, sample in enumerate(samples):
        step_features, frame_features = sample[f"{object}_features"], sample["frame_features"]

        # Used for YouCook2, where text descriptions are unique for each step
        if all_classes_distinct:
            n_samples = sample[f"{object}_ids"].shape[0]
            step_ids = torch.arange(global_step_id_count, global_step_id_count + n_samples)
            global_step_id_count += n_samples
        else:
            step_ids = sample[f"{object}_ids"]

        if distractors is not None:
            bg_step_id = torch.tensor([99999]).to(step_ids.dtype).to(step_ids.device)
            if bg_scope == "class":
                bg_step_id = bg_step_id + sample["cls"]
            if bg_scope == "video":
                global_step_id_count += 1
                bg_step_id = bg_step_id + global_step_id_count

            step_ids = torch.cat([step_ids, bg_step_id])
            step_features = torch.cat([step_features, distractors[i][None, :]])

        if l2_normalize:
            step_features = F.normalize(step_features, p=2, dim=1)
            frame_features = F.normalize(frame_features, p=2, dim=1)

        unique_step_labels, unique_idxs = [
            torch.from_numpy(t) for t in np.unique(step_ids.detach().cpu().numpy(), return_index=True)
        ]
        unique_step_features = step_features[unique_idxs]  # size [K, d]
        sim = unique_step_features @ frame_features.T
        frame_weights = F.softmax(sim / frame_gamma, dim=1)  # size [K, N]
        step_pooled_frames = frame_weights @ frame_features  # size [K, d]
        all_pooled_frames.append(step_pooled_frames)
        pooled_frames_labels.append(unique_step_labels)
        all_step_features.append(step_features)
        all_step_labels.append(step_ids)
    all_pooled_frames = torch.cat(all_pooled_frames, dim=0)
    pooled_frames_labels = torch.cat(pooled_frames_labels, dim=0)
    all_step_features = torch.cat(all_step_features, dim=0)
    all_step_labels = torch.cat(all_step_labels, dim=0)
    assert pooled_frames_labels.shape[0] == all_pooled_frames.shape[0], "Shape mismatch occured"

    unique_labels, unique_idxs = [
        torch.from_numpy(t) for t in np.unique(all_step_labels.detach().cpu().numpy(), return_index=True)
    ]
    unique_step_features = all_step_features[unique_idxs]
    N_steps = all_pooled_frames.shape[0]

    # creating the matrix of targets for the MIL-NCE contrastive objective
    xz_label_mat = torch.zeros([N_steps, unique_labels.shape[0]]).to(all_pooled_frames.device)
    for i in range(all_pooled_frames.shape[0]):
        for j in range(unique_labels.shape[0]):
            xz_label_mat[i, j] = pooled_frames_labels[i] == unique_labels[j]

    # reinforcing existing alignment with step descriptors
    xz_loss = mil_nce(all_pooled_frames, unique_step_features, xz_label_mat, gamma=xz_gamma, hard_ratio=xz_hard_ratio)
    return xz_loss


def compute_alignment_loss(
    samples,
    distractors,
    l2_normalize=False,
    drop_cost_type="max",
    dp_algo="DropDTW",
    keep_percentile=1,
    contiguous=True,
    softning="prob",
    gamma_xz=10,
    gamma_min=1,
    aggregate_loss=True,
    object="step",
):
    gamma_xz = 0.1 if l2_normalize else gamma_xz
    gamma_min = 0.1 if l2_normalize else gamma_min

    # do pre-processing
    zx_costs_list = []
    drop_costs_list = []
    for i, sample in enumerate(samples):
        distractor = None if distractors is None else distractors[i]
        zx_costs, drop_costs, _ = compute_all_costs(
            sample, distractor, gamma_xz, drop_cost_type, keep_percentile, l2_normalize=False, object=object
        )
        zx_costs_list.append(zx_costs)
        drop_costs_list.append(drop_costs)

    min_costs = batch_drop_dtw_machine(zx_costs_list, drop_costs_list, gamma_min=gamma_min, contiguous=contiguous)
    dtw_losses = min_costs / len(samples)
    if aggregate_loss:
        return sum(dtw_losses)
    else:
        return dtw_losses


def compute_offline_alignment_loss(
    samples,
    distractors,
    l2_normalize=False,
    keep_percentile=1,
    contiguous=True,
    gamma_xz=10,
    object="step",
    contrast_frames=True,
    use_graph=False,
):
    gamma_xz = 0.1 if l2_normalize else gamma_xz

    # do pre-processing
    total_loss = 0
    for i, sample in enumerate(samples):
        distractor = None if distractors is None else distractors[i]
        sim = sample[f"{object}_features"] @ sample["frame_features"].T
        with torch.no_grad():
            if not use_graph:
                zx_costs, drop_costs, _ = compute_all_costs(
                    sample, distractor, gamma_xz, "logit", keep_percentile, l2_normalize=False, object=object
                )
                zx_costs, drop_costs = zx_costs.cpu().numpy(), drop_costs.cpu().numpy()
                labels = drop_dtw(zx_costs, drop_costs, contiguous=contiguous, return_labels=True) - 1
            else:
                metadag = sample["metagraph"]
                sorted_node_ids = list(lexicographical_topological_sort(metadag))
                idx2node = {idx: node_id for idx, node_id in enumerate(sorted_node_ids)}
                zx_costs, drop_costs = compute_metadag_costs(
                    sample, idx2node, gamma_xz, keep_percentile=keep_percentile, object=object
                )
                zx_costs, drop_costs = zx_costs.cpu().numpy(), drop_costs.cpu().numpy()
                _, labels = metadag2vid(zx_costs, drop_costs, metadag, idx2node)
                orig_labels = np.array(labels)

                # clean labels
                actual_costs = zx_costs[labels, torch.arange(zx_costs.shape[1])]
                good_costs_mask = actual_costs < drop_costs[0]
                good_costs_mask = np.logical_and(labels > -1, good_costs_mask)
                labels[~good_costs_mask] = -1
                labels = orig_labels if (labels == -1).all() else labels

        # creating corresp matrix from labels
        labels = labels.astype(int)
        corresp_matrix = torch.zeros([sample[f"{object}_features"].shape[0], len(labels)]).to(sim.device)
        for i, l in enumerate(labels):
            if l > -1:
                corresp_matrix[l, i] = 1

        good_row_mask = (corresp_matrix == 1).any(1)
        good_col_mask = (corresp_matrix == 1).any(0)

        if contrast_frames:
            # softmax along remaining rows = competition between frames
            row_sim = sim[good_row_mask, :]
            row_corresp = corresp_matrix[good_row_mask, :]
            row_loss = mil_nce(sim=row_sim, correspondance_mat=row_corresp, gamma=gamma_xz)
        else:
            row_loss = 0

        # softmax along remaining columns = competition between steps
        col_sim = sim[:, good_col_mask]
        col_corresp = corresp_matrix[:, good_col_mask]
        col_loss = mil_nce(sim=col_sim.T, correspondance_mat=col_corresp.T, gamma=gamma_xz)

        total_loss += row_loss + col_loss

    return total_loss / len(samples)


def compute_online_alignment_loss(
    samples,
    distractors,
    l2_normalize=False,
    keep_percentile=1,
    contiguous=True,
    gamma_xz=10,
    object="step",
    use_graph=False,
):
    gamma_xz = 0.1 if l2_normalize else gamma_xz

    # do pre-processing
    if not use_graph:
        total_loss = compute_alignment_loss(
            samples,
            None,
            l2_normalize,
            keep_percentile=keep_percentile,
            contiguous=contiguous,
            gamma_zx=gamma_xz,
            object=object,
        )
    else:
        total_loss = 0
        for i, sample in enumerate(samples):
            distractor = None if distractors is None else distractors[i]
            metadag = sample["metagraph"]
            sorted_node_ids = list(lexicographical_topological_sort(metadag))
            idx2node = {idx: node_id for idx, node_id in enumerate(sorted_node_ids)}
            zx_costs, drop_costs = compute_metadag_costs(
                sample, idx2node, gamma_xz, keep_percentile=keep_percentile, do_logsoftmax=True, object=object
            )
            cost, _ = metadag2vid_soft(zx_costs, drop_costs, metadag, idx2node)
            total_loss = total_loss + cost[0] / zx_costs.shape[0]
        total_loss /= len(samples)

    return total_loss
