from kf_working import KF
import unittest
import numpy as np




class TestKF(unittest.TestCase):


   
    def test_predict(self):
        x = 0.2
        v = 2.4
        kf = KF(initial_x=x, initial_v=v, accel=1.2)
        kf.Predict(dt= 0.1)
        self.assertEqual(kf.cov.shape, (2,2))
        self.assertEqual(kf.mean.shape, (2,))

    def test_after_predict_increase_state_uncertainty(self):
        x = 0.2
        v = 2.4
        kf = KF(initial_x=x, initial_v=v, accel=1.2)
        for i in range(20):
            det_before = np.linalg.det(kf.cov)
            kf.Predict(dt= 0.1)
            det_after = np.linalg.det(kf.cov)
            print(det_before, det_after)
    
    def test_update_decrease_state_uncertainty(self):
        x = 0.2
        v = 2.4
        kf = KF(initial_x=x, initial_v=v, accel=1.2)
        det_before = np.linalg.det(kf.cov)
        kf.update(meas_value=0.1, meas_variance=0.01)
        det_after = np.linalg.det(kf.cov)
        self.assertLess(det_after, det_before)
        

