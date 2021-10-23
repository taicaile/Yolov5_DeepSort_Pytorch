# vim: expandtab:ts=4:sw=4
import numpy as np
import scipy.linalg


"""
Table for the 0.95 quantile of the chi-square distribution with N degrees of
freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
function and used as Mahalanobis gating threshold.
"""
chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919}


class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim) # 8x8
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        # [[ 1   0   0   0   1   0   0   0]
        #  [ 0   1   0   0   0   1   0   0]
        #  [ 0   0   1   0   0   0   1   0]
        #  [ 0   0   0   1   0   0   0   1]
        #  [ 0   0   0   0   1   0   0   0]
        #  [ 0   0   0   0   0   1   0   0]
        #  [ 0   0   0   0   0   0   1   0]
        #  [ 0   0   0   0   0   0   0   1]]

            
        self._update_mat = np.eye(ndim, 2 * ndim) # 4*8
        # array([[1., 0., 0., 0., 0., 0., 0., 0.],
        #        [0., 1., 0., 0., 0., 0., 0., 0.],
        #        [0., 0., 1., 0., 0., 0., 0., 0.],
        #        [0., 0., 0., 1., 0., 0., 0., 0.]])
        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        此函数用于初始化状态state，和协方差矩阵covariance
        """

        mean_pos = measurement # (x,y,a,h)
        mean_vel = np.zeros_like(mean_pos) # 初始化 x,y,a,h 单位时间内的变化量
        mean = np.r_[mean_pos, mean_vel] # 合并，[x,y,a,h,vx,vy,va,vh], (8,) 一维数组
        # 初始化一个标准差
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std)) # 协方差, 8*8

        return mean, covariance

    def predict(self, mean, covariance):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        根据过去的状态预测当前的状态，注意这里没有 measurement 输入
        这里的输入mean，会包含 [x,y,a,h] 的变化量，这里并不是通过差值计算获得
        """

        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]]
        # 这里的motion_cov 作为噪声矩阵
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        # x' = Fx, (8x8)x(8x1)
        mean = np.dot(self._motion_mat, mean)
        # P' = FPF(T) + Q, 这里的covariance添加了噪声
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov
        # 返回预测的 状态向量，和协方差矩阵
        return mean, covariance

    def project(self, mean, covariance):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std)) # 初始化噪声矩阵R

        mean = np.dot(self._update_mat, mean) # 将均值向量映射到检测空间，即Hx', (4x8)(8x1) = 4x1
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        # 将协方差矩阵映射到检测空间，即HP'H^T
        # (4x8)(8x8)(8x4) = 4x4
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.
        根据 预测的 mean， variance，和measurement测量值，计算卡尔曼增益，获取融合后的结果
        """
        # 将mean和covariance映射到检测空间
        projected_mean, projected_cov = self.project(mean, covariance)
        # 这里的projected_mean,projected_cov 的size都是 4

        #  计算卡尔曼增益K, 8x4
        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        # array([[    0.86777,           0,           0,           0],
        #        [          0,     0.86777,           0,           0],
        #        [          0,           0,    0.019608,           0],
        #        [          0,           0,           0,     0.86777],
        #        [    0.20661,           0,           0,           0],
        #        [          0,     0.20661,           0,           0],
        #        [          0,           0,  9.8039e-09,           0],
        #        [          0,           0,           0,     0.20661]])

        # z - Hx', innovation.shape=(4,), 不包含变化量， 代表测量值和预测值的差值
        innovation = measurement - projected_mean
        #TODO  这里没有看懂
        # x = x' + Ky, mean.shape=(8,), innovation.shape=(4,), kalman_gain.shape=(8,4)
        # mean是 (8,) 1D array, 
        new_mean = mean + np.dot(innovation, kalman_gain.T) # (8,)
        # P = (I - KH)P'
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements,
                        only_position=False):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
