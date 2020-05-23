import numpy as np
import pandas as pd


class KalmanFilter(object):
    def __init__(self, a, b, c, r_ww, r_vv, x_t_t_ini=None, p_t_t_ini=None):
        """

        :param a:
        :param b:
        :param c:
        :param r_ww:
        :param r_vv:
        :param x_t_t_ini:
        :param p_t_t_ini:
        """
        self.a = a
        self.b = b
        self.c = c

        self.r_ww = r_ww
        self.r_vv = r_vv

        self.x_t_t1 = None
        self.p_t_t1 = None

        self.x_t_t = x_t_t_ini
        self.p_t_t = p_t_t_ini

        self.e_t = None
        self.r_ee = None
        self.inv_r_ee = None

        self.k_t = None

        self.history = []
        self.t = 0

    @staticmethod
    def expand_cols(name, vec):
        """

        :param name:
        :param vec:
        :return:
        """
        vec = vec.ravel()
        n = int(vec.shape[0])
        names = [f"{name}.{i+1}" for i in range(n)]
        d = dict(zip(names, vec))
        return d

    def update_history(self):
        """

        :return:
        """
        self.t += 1
        state = {'t': self.t}
        state.update(self.expand_cols('x_t_t1', self.x_t_t1))
        state.update(self.expand_cols('p_t_t1', self.p_t_t1))
        state.update(self.expand_cols('x_t_t', self.x_t_t))
        state.update(self.expand_cols('p_t_t', self.p_t_t))
        state.update(self.expand_cols('k_t', self.k_t))
        self.history.append(state)

    @staticmethod
    def _dot_3(x1, x2, x3):
        """

        :param x1:
        :param x2:
        :param x3:
        :return:
        """
        return np.dot(np.dot(x1, x2), x3)

    def _prediction(self, u):
        """

        :param u:
        :return:
        """
        self.x_t_t1 = np.dot(self.a, self.x_t_t) + np.dot(self.b, u)
        self.p_t_t1 = self._dot_3(self.a, self.p_t_t, self.a.T) + self.r_ww

    def _innovation(self, y_t):
        """

        :param y_t:
        :return:
        """
        self.e_t = y_t - np.dot(self.c, self.x_t_t1)
        self.r_ee = self._dot_3(self.c, self.p_t_t1, self.c.T) + self.r_vv
        self.inv_r_ee = np.linalg.inv(self.r_ee)

    def _gain(self):
        """

        :return:
        """
        self.k_t = self._dot_3(self.p_t_t1, self.c.T, self.inv_r_ee)

    def _update(self):
        """

        :return:
        """
        self.x_t_t = self.x_t_t1 + np.dot(self.k_t, self.e_t)
        self.p_t_t = np.dot(np.identity(self.x_t_t.shape[0]) - np.dot(self.k_t, self.c), self.p_t_t1)

    def step(self, u, y_t):
        """

        :param u:
        :param y_t:
        :return:
        """
        self._prediction(u)
        self._innovation(y_t)
        self._gain()
        self._update()
        self.update_history()
        return self.x_t_t

    def get_history(self):
        """

        :return:
        """
        return pd.DataFrame(self.history)


if __name__ == '__main__':
    _dim_x = 1
    _dim_y = 1
    _dim_u = 1

    _r_ww = np.diag([0.0001]*_dim_x)
    _r_vv = np.diag([4]*_dim_y)

    _a = 0.97 * np.identity(_dim_x)
    _b = 100 * np.ones((_dim_x, _dim_u))
    _c = 2 * np.ones((_dim_y, _dim_x))

    _u = 300 * 1e-6 * np.ones((_dim_u, 1))
    _x_t_t_ini = 2.5*np.ones((_dim_x, 1))
    _p_t_t_ini = np.diag([0.1]*_dim_x)
    _y_t = 2.1

    kf = KalmanFilter(_a, _b, _c, _r_ww, _r_vv, x_t_t_ini=_x_t_t_ini, p_t_t_ini=_p_t_t_ini)
    kf.step(u=_u, y_t=_y_t)
    print()
