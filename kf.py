
from tkinter import SEL
import numpy as np
import os
#from zed_odom import 
#offsets of each variable in the state vector

txt = "zed_odom.txt"
folder = os.path.dirname(os.path.abspath(__file__))
file = os.path.join(folder, txt)
               
zed_odom = open(file,"r")
blank = []
dt = np.array(blank)

Data = open(file,"r")
for line in Data:
    secs = line.find("secs:")
    if secs != -1:
        dt=np.append(dt,float(line[secs+6:].strip())/1000000000)
               
        

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
    xv = (x_pose[i+1]-x_pose[i])/dt[i]
    x_vel = np.append(x_vel, xv)
    yv = (y_pose[i+1]-y_pose[i])/dt[i]
    y_vel = np.append(y_vel, yv)
    zv = (z_pose[i+1]-z_pose[i])/dt[i]
    z_vel = np.append(z_vel, zv)
       

x_accel = np.array(0)
y_accel =np.array(0)
z_accel =np.array(0)
len_vel = int(np.shape(x_vel)[0])

for i in range(len_vel -1):
    xa = (x_vel[i+1]-x_vel[i])/dt[i]
    x_accel = np.append(x_accel, xa)
    ya = (y_vel[i+1]-y_vel[i])/dt[i]
    y_accel = np.append(y_accel, ya)
    za = (z_vel[i+1]-z_vel[i])/dt[i]
    z_accel = np.append(z_accel, za)

      

class KF:
    all_Xs = np.array(blank)
    def Start(Dimension:int):
        a = 0
        
        X_initial = x_pose[a]
        vX_initial = x_vel[a]
        Y_initial = y_pose[a]
        vY_initial = y_vel[a]
        Z_initial = z_pose[a]
        vZ_initial = z_vel[a]

        var_x = np.var(x_pose)
        var_vx = np.var(x_vel)
        var_y = np.var(y_pose)
        var_vy = np.var(y_vel)
        var_z = np.var(z_pose)
        var_vz = np.var(z_vel)
        
        if Dimension == 2:

            X = np.array([X_initial,Y_initial,vX_initial,vY_initial]).reshape(4,1)
            A = np.array([[1,0,dt[a],0],[0,1,0,dt[a]],[0,0,1,0],[0,0,0,1]])
            u = np.array([x_accel[0],y_accel[0]]).reshape(2,1)
            P = np.array([[var_x**2,0,0,0],[0,var_y**2,0,0],[0,0,var_vx**2,0],[0,0,0,var_vy**2]])
            B = np.array([[0.5 *  dt[a]**2, 0], [0,0.5 * dt[a]**2],[dt[a], 0],[dt[a], 0]])

        if Dimension == 3:
            X = np.array([X_initial,Y_initial,Z_initial,vX_initial,vY_initial,vZ_initial]).reshape(6,1)
            A = np.array([[1,0,0,dt[a],0,0],[0,1,0,0,dt[a],0],[0,0,1,0,0,dt[a]],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            u = np.array([x_accel[a],y_accel[a],z_accel[a]]).reshape(3,1)
            P = np.array([[var_x**2,0,0,0,0,0],[0,var_y**2,0,0,0,0],[0,0,var_z**2,0,0,0],[0,0,0,var_vx**2,0,0],[0,0,0,0,var_vy**2,0],[0,0,0,0,0,var_vz**2]])
            B = np.array([[0.5 *  dt[a]**2, 0,0], [0,0.5 * dt[a]**2,0],[0,0,0.5 * dt[a]**2],[dt[a], 0,0],[0,dt[a], 0],[0,0,dt[a]]])

            

        new_x = A.dot(X)+ B.dot(u)
        new_P = A.dot(P).dot(A.T)
        P = new_P
        X = new_x
        A = [X,P]
        return A
    def Predict(i, Dimension:int) -> None:
        # X(kp) = Ax(k-1)+Bu+ wk
        # P(kp) = AP(k-1)AT +Qk
        
        X = KF.Start(Dimension)[0]
        P = KF.Start(Dimension)[1]
        X_i = x_pose[i]
        vX_i = x_vel[i]
        Y_i = y_pose[i]
        vY_i = y_vel[i]
        Z_i = z_pose[i]
        vZ_i = z_vel[i]
        if Dimension == 2:
            A = np.array([[1,0,dt[i],0],[0,1,0,dt[i]],[0,0,1,0],[0,0,0,1]])
            u = np.array([x_accel[0],y_accel[0]]).reshape(2,1)      
            B = np.array([[0.5 *  dt[i]**2, 0], [0,0.5 * dt[i]**2],[dt[i], 0],[dt[i], 0]])
          
             
                    # y = cX(k) + zk
            # K = P(kp)H / HP(kp)HT+r
            # x = x(kp) + K(Y- HK(p))
            # P = (I - KH)P(kp)

            Xy = np.array([X_i,Y_i,vX_i,vY_i]).reshape(4,1)
            H = np.eye(4)  
            C = np.eye(4)
            Y = C.dot(Xy)
            I = np.eye(4)
           
        
        if Dimension == 3:
        
            A = np.array([[1,0,0,dt[i],0,0],[0,1,0,0,dt[i],0],[0,0,1,0,0,dt[i]],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]])
            u = np.array([x_accel[i],y_accel[i],z_accel[i]]).reshape(3,1)
            B = np.array([[0.5 *  dt[i]**2, 0,0], [0,0.5 * dt[i]**2,0],[0,0,0.5 * dt[i]**2],[dt[i], 0,0],[0,dt[i], 0],[0,0,dt[i]]])
           
            
            Xy = np.array([X_i,Y_i,Z_i,vX_i,vY_i,vZ_i]).reshape(6,1)
            H = np.eye(6)  
            C = np.eye(6)
            Y = C.dot(Xy)
            I = np.eye(6)


    


        

        Y = C.dot(Xy)
       
        PHT = P.dot(H.T)

        # S = HPH' + R
        # project system uncertainty into measurement space
        S = H.dot(PHT)
        SI =np.linalg.inv(S)
        # K = PH'inv(S)
        # map system uncertainty into kalman gain
        K = PHT.dot(SI)
        new1_x = X + K.dot(Y- (H.dot(Y)))
        new_P = (I - K.dot(H)).dot(P)

        P = new_P
        X = new1_x
        KF.all_Xs = np.append(KF.all_Xs, X)


KF.Start(2)
for i in range(len(x_pose)):
    KF.Predict(i,2)
print(KF.all_Xs)


                



