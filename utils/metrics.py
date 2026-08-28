import numpy as np


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    d += 1e-12
    return 0.01 * (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    rse = RSE(pred, true)
    corr = CORR(pred, true)

    return mae, mse, rmse, mape, mspe, rse, corr


def instance_averaged_metrics(pred, true, instance_idx):
    # Calculate metrics for each unique instance, then average across instances.
    unique_instance_idxs = np.unique(instance_idx)
    maes, mses, rmses, mapes, mspes, rses, corrs = [], [], [], [], [], [], []
    for l in unique_instance_idxs:
        idx = instance_idx == l
        mae, mse, rmse, mape, mspe, rse, corr = metric(pred[idx], true[idx])
        maes.append(mae)
        mses.append(mse)
        rmses.append(rmse)
        mapes.append(mape)
        mspes.append(mspe)
        rses.append(rse)
        corrs.append(corr)
    return (
        np.mean(maes),
        np.mean(mses),
        np.mean(rmses),
        np.mean(mapes),
        np.mean(mspes),
        np.mean(rses),
        np.mean(corrs),
    )
