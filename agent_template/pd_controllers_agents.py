import numpy as np
import mujoco as mj
from scipy.linalg import cho_solve, cho_factor
def compute_acceleration(q_error,qdot,C,tau_ext,new_mass,KP,KD):
    
    # we add none on the indexing to add a dimension and avoid errors
    # add a dimension to vectors to fit matrix operation requirements
    C = C[:, None]
    q_error = q_error[:, None]
    qdot = qdot[:, None]
    tau_ext = tau_ext[:,None]
    
    # Calculate the proportional and damping forces
    prop_force = KP @ q_error
    damp_force = KD @ qdot
    
    # Combine forces for the equation
    combined_forces = -C - prop_force - damp_force + tau_ext
      
    #facotor the mass, to solve the equation    
    chol_factor, lower = cho_factor(new_mass, overwrite_a=True, check_finite=False)
    
    # solve the equation
    #we can also solve it with this line of code, but I will use the cholesky way
    #qdot_dot = jp.linalg.solve(new_mass, combined_forces)
    
    qdot_dot = cho_solve((chol_factor, lower), combined_forces, overwrite_b=True, check_finite=False)
    #return it back to one dim array
    return qdot_dot.squeeze()
    


# def calculate_mass_check_invertible(M,KD,dt,time):
    
    
#     #print(M)
    
#     new_mass = M + (KD * dt)
    
#     # add regularization at the begging to avoid error with Nan
#     if time == 1:
#         # regularization is called Tikhonov
#         epsilon = 1e-6  # Small regularization term
#         new_mass = new_mass + epsilon * jp.eye(new_mass.shape[0])
    
    
#     # Calculate the condition number
#     cond_number = jp.linalg.cond(new_mass)

#     # Now check the condition number
#     if cond_number < 1/jp.finfo(new_mass.dtype).eps:
#         # It is likely invertible and well-conditioned
#         pass
#     else:
#         # The matrix may be singular or near-singular
#         print(f"Warning: High condition number ({cond_number}) indicates potential numerical instability. current time {time}")

#     return new_mass

def init_corolis_mass_external(mjModel,mjData,kp_array,kd_array):
    #get the centrifugal force
    #dim (nv)
    C = mjData.qfrc_bias.copy()
    
    tau_ext = mjData.qfrc_applied.copy()  # This should represent the external forces

    #get the mass matrix fore the inertia
    M = np.zeros((mjModel.nv,mjModel.nv))
    mj.mj_fullM(mjModel, M, mjData.qM)
    #get the diagonal matrices
    KP = np.diag(kp_array)
    KD = np.diag(kd_array)
    
    return C,M,KP,KD, tau_ext

def calculate_new_mass(M,KD,dt):
    #jax.debug.print("M: {}", M)
    
    new_mass = M + (KD * dt)

    return new_mass
def stable_pd_controller(target,mjModel,mjData,q,qdot,kp_array,kd_array,dt,time):
    
    q = q.copy()
    qdot = qdot.copy()
    
    target_q_next= target
    
    error_q_next =(q[7:] + (qdot[6:]*dt) )-target_q_next
    
    
    #first reshape the kp and kd by adding elemnts at the beginning 6
    #remember the size of the kp and kd is 28, and we want 34 to match
    #the dofs
    kp_matrix = np.concatenate((np.zeros(6),kp_array))
    kd_matrix = np.concatenate((np.zeros(6),kd_array))
    
    angular_error = qdot
    error_pos = np.concatenate((np.zeros(6), error_q_next))
    
    
    #initialize the variables for getting the acceleration
    C,M,KP,KD,tau_ext=init_corolis_mass_external(mjModel, mjData,kp_matrix,kd_matrix)
    #get the mass inertia matrix with the added kd dy
    new_mass = calculate_new_mass(M,KD,dt)

    #calculate the predicted acceleration      
    qdot_dot = compute_acceleration(error_pos,angular_error,C,tau_ext,new_mass,KP,KD)
    
    # add the predicted error for the qdot, and then add than on the principal equation
    angular_error = angular_error + (qdot_dot*dt)
    #now get the torque avoiding the root
    tau = -kp_array * error_pos[6:] - kd_array * angular_error[6:]
    return tau