from sklearn.metrics import v_measure_score, adjusted_rand_score, accuracy_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
import numpy as np
import torch

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

def inference(loader, model, device, view, data_size, feature_mode='common'):
    if feature_mode not in ('common', 'concat'):
        raise ValueError('feature_mode must be common or concat')
    model.eval()
    commonZ = []
    labels_vector = []
    for step, (xs, y, _) in enumerate(loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        with torch.no_grad():
            commonz, _ = model.TMCNF(xs)
            if feature_mode == 'concat':
                # Use exactly the projected hs from TMCN, in view order.
                # These are not guaranteed to be disentangled specific factors.
                _, _, hs = model(xs)
                commonz = torch.cat([commonz, *hs], dim=1)
            commonz = commonz.detach()
            commonZ.extend(commonz.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
    labels_vector = np.array(labels_vector).reshape(data_size)
    commonZ = np.array(commonZ)
    return labels_vector, commonZ

def valid(model, device, dataset, view, data_size, class_num, seed=10,
          feature_mode='common'):
    test_loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )
    labels_vector, commonZ = inference(test_loader, model, device, view, data_size, feature_mode)
    print('---------train over---------')
    print('Clustering results: mode={}, dimensions={}'.format(feature_mode, commonZ.shape[1]))
    kmeans = KMeans(n_clusters=class_num, n_init=100, random_state=seed)
    y_pred = kmeans.fit_predict(commonZ)
    nmi, ari, acc, pur = evaluate(labels_vector, y_pred)
    print('ACC = {:.4f} NMI = {:.4f} PUR={:.4f} ARI = {:.4f}'.format(acc, nmi, pur, ari))


    return dict(ACC=float(acc), NMI=float(nmi), PUR=float(pur), ARI=float(ari))
