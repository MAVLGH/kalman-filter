from kalman_filter import KalmanFilter
from model import DynamicalSystem

import numpy as np

_dim_x = 1
_dim_y = 1
_dim_u = 1

_r_ww = np.diag([0.0001] * _dim_x)
_r_vv = np.diag([4] * _dim_y)

_a = 0.97 * np.identity(_dim_x)
_b = 100 * np.ones((_dim_x, _dim_u))
_c = 2 * np.ones((_dim_y, _dim_x))

_u = 300 * 1e-6 * np.ones((_dim_u, 1))
_x_ini = 2.5 * np.ones((_dim_x, 1))
_x_t_t_ini = 3 * np.ones((_dim_x, 1))
_p_t_t_ini = np.diag([2] * _dim_x)


ds = DynamicalSystem(_a, _b, _c, _r_ww, _r_vv, x_ini=_x_ini)
kf = KalmanFilter(_a, _b, _c, _r_ww, _r_vv, x_t_t_ini=_x_t_t_ini, p_t_t_ini=_p_t_t_ini)

x_t_t = []
for i in range(100):
    _y_t = ds.y_step(u=_u)
    x_tt = kf.step(u=_u, y_t=_y_t)
    x_t_t.append(x_tt)

df_history = ds.get_history()
df_history.loc[:, 'x_t_t'] = np.array(x_t_t)[:, :, 0]
print()

