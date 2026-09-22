from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch
from features import fusion_features, validate_alpha

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)

def evaluate(label, pred):
    nmi = v_measure_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    return nmi, ari, acc, pur

def inference(loader, model, device, view, data_size, feature_mode='common', fusion_alpha=None,
              view_index=None):
    if feature_mode not in ('common', 'concat', 'weighted_concat', 'complement', 'view'):
        raise ValueError('feature_mode must be common, concat, weighted_concat, complement, or view')
    if feature_mode == 'view':
        if not isinstance(view_index, int) or not 0 <= view_index < view:
            raise ValueError('view mode requires a valid zero-based view_index')
    elif view_index is not None:
        raise ValueError('view_index applies only to view mode')
    if fusion_alpha is not None:
        validate_alpha(fusion_alpha)
        if feature_mode != 'weighted_concat':
            raise ValueError('fusion_alpha applies only to weighted_concat')
    if feature_mode == 'complement' and not getattr(model, 'complementary', False):
        raise ValueError('complement mode requires a checkpoint trained with --complementary')
    model.eval()
    commonZ = []
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            if feature_mode == 'view':
                _, _, hs = model(xs)
                commonz = hs[view_index]
            else:
                commonz, _ = model.TMCNF(xs)
            if feature_mode in ('concat', 'weighted_concat'):
                # Use exactly the projected hs from TMCN, in view order.
                # These are not guaranteed to be disentangled specific factors.
                _, _, hs = model(xs)
                commonz = fusion_features(commonz, hs, feature_mode, fusion_alpha)
            elif feature_mode == 'complement':
                _, zs, _ = model(xs)
                specs = model.complementary_features(zs)
                commonz = torch.cat([commonz, *specs], dim=1)
            commonz = commonz.detach()
            commonZ.extend(commonz.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    commonZ = np.array(commonZ)
    return labels_vector, commonZ

def valid(model, device, dataset, view, data_size, class_num, seed=10,
          feature_mode='common', fusion_alpha=None, view_index=None):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, commonZ = inference(test_loader, model, device, view, data_size,
                                      feature_mode, fusion_alpha, view_index)
    print('---------train over---------')
    label = 'view_{}'.format(view_index + 1) if feature_mode == 'view' else feature_mode
    print('Clustering results: mode={}, dimensions={}'.format(label, commonZ.shape[1]))
    if feature_mode == 'weighted_concat':
        print('Shared distance weight alpha={}'.format(1.0 / (view + 1) if fusion_alpha is None else fusion_alpha))
    kmeans = KMeans(n_clusters=class_num, n_init=100, random_state=seed)
    y_pred = kmeans.fit_predict(commonZ)
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))


    return dict(ACC=float(acc), NMI=float(nmi), PUR=float(pur), ARI=float(ari))
