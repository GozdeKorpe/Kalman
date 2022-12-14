from tkinter import SEL
import numpy as np


#offsets of each variable in the state vector
iX = 0
iV = 1
NUMVARS = iV + 1

class KF:
    def __init__(self, initial_x: float,
                       initial_v:float, 
                       accel:float) -> None:
        #mean of state
        self._x = np.zeros(NUMVARS)
        self._x[iX] = initial_x
        self._x[iV] = initial_v
        
        self._accel = accel

        #covariance of state
        self._P = np.eye(NUMVARS)

    def Predict(self, dt: float) -> None:
        # X(k+1) = F * x(k)
        # P(k+1) = F * P(K)*Ft + G*a*Gt
        F = np.array([[1,dt],[0,1]])
        new_x = F.dot(self._x)
        G = np.array([0.5 * dt**2, dt]).reshape((2,1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T)* self._accel
        self._P = new_P
        self._x = new_x

    def update(self, meas_value: float, meas_variance: float):
        # y = z - H*x
        # K = P * Ht * S^-1
        # S = H * P * Ht + R
        # x = x(k) + K * y
        # P = (I - K * H)*P(k)
        
        H = np.array([1, 0]).reshape((1,2))
        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R
        K = self._P.dot(H.T).dot(np.linalg.inv(S))
        new_x = self._x + K.dot(y)
        I = np.eye(2)
        new_P = (I - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x

    @property
    def pos(self) -> float:
        return self._x[0]

    @property
    def vel(self) -> float:
        return self._x[1]

    @property
    def cov(self) -> np.array:
        return self._P

    @property
    def mean(self) -> np.array:
        return self._x