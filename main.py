"""

James V. Candy - Bayesian Signal Processing, Classical, Modern, and Particle Filtering Methods.
Example 5.1 - RC circuit

"""

from kalman_filter import KalmanFilter
from model import DynamicalSystem

import numpy as np
import matplotlib.pyplot as plt


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
_x_t_t_ini = 10 * np.ones((_dim_x, 1))
_p_t_t_ini = np.diag([10] * _dim_x)


ds = DynamicalSystem(_a, _b, _c, _r_ww, _r_vv, x_ini=_x_ini)
kf = KalmanFilter(_a, _b, _c, _r_ww, _r_vv, x_t_t_ini=_x_t_t_ini, p_t_t_ini=_p_t_t_ini)

for i in range(200):
    _y_t = ds.y_step(u=_u)
    _ = kf.step(u=_u, y_t=_y_t)

ds_history = ds.get_history()
kf_history = kf.get_history()

df_history = ds_history.merge(kf_history, on='t')
df_history.loc[:, 'x_e.1'] = df_history['x_t.1'] - df_history['x_t_t.1']
df_history.loc[:, 'I_sup_x.1'] = df_history['x_t_t.1'] + 1.96 * (df_history['p_t_t.1']**0.5)
df_history.loc[:, 'I_inf_x.1'] = df_history['x_t_t.1'] - 1.96 * (df_history['p_t_t.1']**0.5)

fig, ax = plt.subplots()
ax.plot(df_history['t'], df_history['x_t.1'], label='x_t.1', color='r')
ax.plot(df_history['t'], df_history['x_t_t.1'], label='x_t_t.1', color='b')
ax.fill_between(df_history['t'], df_history['I_inf_x.1'], df_history['I_sup_x.1'], color='b', alpha=.1)
plt.legend()
plt.show()

print()

