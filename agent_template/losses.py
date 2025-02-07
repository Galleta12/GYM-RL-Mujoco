import jax
from jax import numpy as jp
from some_math.rotation6D import quaternion_to_rotation_6d


import sys
import os
current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


#got this from brax
#global space
def loss_l2_relpos(pos, ref_pos):
    relpos, ref_relpos = (pos - pos[0])[1:], (ref_pos - ref_pos[0])[1:]
    relpos_loss = (((relpos - ref_relpos) ** 2).sum(-1) ** 0.5).mean()
        
    #relpos_loss = relpos_loss
        
    return relpos_loss


def loss_l2_pos(pos, ref_pos):
    pos_loss = (((pos - ref_pos) ** 2).sum(-1)**0.5).mean()
    return pos_loss

#position in global
def mse_pos(pos, ref_pos):
    #jax.debug.print("global pos: {}", pos)
    #jax.debug.print("ref pos: {}", ref_pos)
    
    pos_loss = ((pos - ref_pos) ** 2).sum(-1).mean()
    
    
    #jax.debug.print("mse_pos: {}", pos_loss)
    
    return pos_loss

#rotation this is already in 6D form
def mse_rot(rot, ref_rot):
    rot_loss = ((rot - ref_rot) ** 2).sum(-1).mean()
    #jax.debug.print("mse_rot: {}", rot_loss)
    
    return rot_loss

def mse_vel(vel, ref_vel):
    vel_loss = ((vel - ref_vel) ** 2).sum(-1).mean()
    
    #jax.debug.print("vel loss: {}", vel_loss)
    
    return vel_loss

def mse_ang(ang, ref_ang):
    ang_loss = ((ang - ref_ang) ** 2).sum(-1).mean()
    
    #jax.debug.print("ang_loss: {}", ang_loss)
    
    return ang_loss
