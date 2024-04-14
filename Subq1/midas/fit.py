import numpy as np

def ssr(a, x, y, yl, weight_method):
    xw, _ = weight_method.x_weighted(x, a[2:4])
    error = y - a[0] - a[1] * xw
    if yl is not None:
        error -= np.dot(yl, a[4:])
    return error


def ssr2(a, x1, x2, y, yl, weight_method1, weight_method2):
    xw1, _ = weight_method1.x_weighted(x1, a[3:5])
    xw2, _ = weight_method2.x_weighted(x2, a[5:7])
    error = y - a[0] - a[1] * xw1- a[2] * xw2
    if yl is not None:
        error -= np.dot(yl, a[7:])
    return error


def jacobian(a, x, y, yl, weight_method):
    jwx = jacobian_wx(x, a[2:4], weight_method)
    xw, _ = weight_method.x_weighted(x, a[2:4])
    if yl is None:
        jac_e = np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), (a[1] * jwx)], axis=1)
    else:
        jac_e = np.concatenate([np.ones((len(xw), 1)), xw.reshape((len(xw), 1)), (a[1] * jwx), yl], axis=1)
    return -1.0 * jac_e


def jacobian2(a, x1, x2, y, yl, weight_method1, weight_method2):
    jwx1 = jacobian_wx(x1, a[3:5], weight_method1)
    jwx2 = jacobian_wx(x2, a[5:7], weight_method2)
    xw1, _ = weight_method1.x_weighted(x1, a[3:5])
    xw2, _ = weight_method2.x_weighted(x2, a[5:7])
    if yl is None:
        jac_e = np.concatenate([np.ones((len(xw1), 1)), xw1.reshape((len(xw1), 1)), xw2.reshape((len(xw2), 1)), (a[1] * jwx1), (a[2] * jwx2)], axis=1)#-jacobian
    else:
        jac_e = np.concatenate([np.ones((len(xw1), 1)), xw1.reshape((len(xw1), 1)), xw2.reshape((len(xw2), 1)), (a[1] * jwx1), (a[2] * jwx2), yl], axis=1)#-jacobian
    return -1.0 * jac_e


def jacobian_wx(x, params, weight_method):
    eps = 1e-6
    jt = []
    for i, p in enumerate(params):
        dp = np.concatenate([params[0:i], [p + eps / 2], params[i + 1:]])
        dm = np.concatenate([params[0:i], [p - eps / 2], params[i + 1:]])
        jtp, _ = weight_method.x_weighted(x, dp)
        jtm, _ = weight_method.x_weighted(x, dm)
        jt.append((jtp - jtm) / eps)
    return np.column_stack(jt)
