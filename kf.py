from tkinter import SEL
import numpy as np
import os
#from zed_odom import Data

#offsets of each variable in the state vector

class Data:
    blank = []
    x_pose =np.array(blank)
        

    def Cordinates(txt:str):
        folder = os.path.dirname(os.path.abspath(__file__))
        file = os.path.join(folder, txt)
               
        zed_odom = open(file,"r")
        blank = []

        data = open(file,"r")
        for line in data:
            secs = line.find("secs:")
            if secs != -1:
                dt = float(line[secs+6:].strip())
               
                break

        x_cor = np.array(blank)
        
        y_cor = np.array(blank)
        z_cor = np.array(blank)
        for line in zed_odom:
            

            x_first = line.find("x:")
            y_first = line.find("y:")
            z_first = line.find("z:")
            z_end = line.find("orientation:") 
            
            if x_first != -1:
                
                x = line[x_first+2:y_first].strip()
                
                x_cor = np.append(x_cor,float(x))           
                
                
                
            
            if y_first != -1:
                y = line[y_first+2:z_first].strip()
       
                y_cor = np.append(y_cor,float(y))
            if z_first != -1:
                z = line[z_first+2:z_end].strip()
                z_cor = np.append(z_cor,float(z))
            
            
        
        len_cor = int(np.shape(x_cor)[0])
        
       
        zed_odom.close()
        
        x_pose =np.array(blank)
        y_pose =np.array(blank)
        z_pose =np.array(blank)
        for i in range(int(len_cor/4)):
            x_pose = np.append(x_pose, x_cor[4*i])
            y_pose = np.append(y_pose, y_cor[4*i])
            z_pose = np.append(z_pose, z_cor[4*i])
      

        x_vel = np.array(0)
        y_vel =np.array(0)
        z_vel =np.array(0)
        len_pose = int(np.shape(x_pose)[0])

        for i in range(len_pose -1):
            xv = (x_pose[i+1]-x_pose[i])/dt
            x_vel = np.append(x_vel, xv)
            yv = (y_pose[i+1]-y_pose[i])/dt
            y_vel = np.append(y_vel, yv)
            zv = (z_pose[i+1]-z_pose[i])/dt
            z_vel = np.append(z_vel, zv)
       

        x_accel = np.array(0)
        y_accel =np.array(0)
        z_accel =np.array(0)
        len_vel = int(np.shape(x_vel)[0])

        for i in range(len_vel -1):
            xa = (x_vel[i+1]-x_vel[i])/dt
            x_accel = np.append(x_accel, xa)
            ya = (y_vel[i+1]-y_vel[i])/dt
            y_accel = np.append(y_accel, ya)
            za = (z_vel[i+1]-z_vel[i])/dt
            z_accel = np.append(z_accel, za)
        
        return x_pose, y_pose,z_pose,x_vel,y_vel,z_vel,x_accel,y_accel,z_accel

class KF:

    data = Data.Cordinates("zed_odom.txt")
    def Predict() -> None:
        # X(kp) = Ax(k-1)+Bu+ wk
        # P(kp) = AP(k-1)AT +Qk
        a = 0
        X_initial = Data.x_pose[a]
        vX_initial = Data.x_vel[a]
        Y_initial = Data.y_pose[a]
        vY_initial = Data.y_vel[a]
        X = np.array([X_initial],[Y_initial],[vX_initial],[vY_initial])
        A = np.array([[1,0,Data.dt,0],[0,1,0,Data.dt],[0,0,1,0],[0,0,0,1]])
        u = np.array([Data.x_accel[0]],[Data.y_accel[0]])
        var_x = np.var(Data.x_pose)
        var_vx = np.var(Data.x_vel)
        var_y = np.var(Data.y_pose)
        var_vy = np.var(Data.y_vel)
        P = np.array([var_x**2,0,0,0],[0,var_y**2,0,0],[0,0,var_vx**2,0],[0,0,0,var_vy**2])
        B = np.array([0.5 * Data. dt**2, 0], [0,0.5 * Data.dt**2],[Data.dt, 0],[Data.dt, 0])
        new_x = A.dot(X)+ B.dot(u)
        new_P = A.dot(P).dot(A.T) 
        P = new_P
        X = new_x

    def Update():
        # y = cX(k) + zk
        # K = P(kp)H / HP(kp)HT+r
        # x = x(kp) + K(Y- HK(p))
        # P = (I - KH)P(kp)
        
        H = np.eye(4)  
        C = np.eye(4)
        Y = C.dot(X)
        r = np.array()
        K = (P.dot(H))/ H.dot(P).dot(H.T) + r
        new_x = X + K.dot(Y-H.dot(Y))
        I = np.eye(4)
        new_P = (I - K.dot(H)).dot(P)

        P = new_P
        X = new_x

Data.Cordinates("zed_odom.txt")
KF.Predict()

