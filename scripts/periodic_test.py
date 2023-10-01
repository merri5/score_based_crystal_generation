from matplotlib import pyplot as plt
import numpy as np
import torch

# import matplotlib
# matplotlib.rc('text', usetex=True)
# matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

def periodic_distance(f1, f2):
    f1_ = 2*torch.pi*f1
    f2_ = 2*torch.pi*f2
    # zero = torch.tensor([0.])
    zero = torch.zeros_like(f1_)
    # print(torch.exp(torch.complex(zero,f1_)))
    # print(torch.exp(torch.complex(zero,-f2_)))
    return torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
    
def periodic_distance_3D(f1, f2):
    f1_ = 2*torch.pi*f1
    f2_ = 2*torch.pi*f2
    zero = torch.tensor([0.])
    zero = torch.zeros_like(f1_)
    # print(torch.exp(torch.complex(zero,f1_)))
    # print(torch.exp(torch.complex(zero,-f2_)))
    # distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
    distance = torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) )
    # print(distance)
    distance_vec = (-distance + 1)/2 
    # print(distance_vec)
    return torch.matmul(distance_vec,distance_vec)

def periodic_distance_3D_loss(f1,f2):
    
        f2 = - f2
        f1_ = 2*torch.pi*f1
        f2_ = 2*torch.pi*f2
        zero = torch.tensor([0.]) #, device=f1_.device)
        # print(torch.exp(torch.complex(zero,f1_)))
        # print(torch.exp(torch.complex(zero,-f2_)))
        distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
        distance_vec = (-distance + 1)/2 
        # print(distance_vec)

        # print(distance_vec)
        # loss_per_atom = 0.5 *  torch.sum((distance_vec)**2, dim=1) * used_sigmas_per_atom**2
        # loss_per_atom = torch.sum((distance_vec)**2, dim=1)
        return torch.sum(distance_vec)

def mse_distance(f1,f2):
    loss_per_atom = torch.square(f1 + f2)
    loss_per_atom = 0.5 * loss_per_atom
    return loss_per_atom


def mse_distance_modified(f1,f2):
    loss_per_atom = torch.square((f1 + f2 + 0.5)%1. - 0.5)
    loss_per_atom = loss_per_atom
    return loss_per_atom



def periodic_distance_3D_loss_modified(f1,f2):
    
        f2 = - f2
        f1_ = 2*torch.pi*f1
        f2_ = 2*torch.pi*f2
        zero = torch.tensor([0.]) #, device=f1_.device)
        # print(torch.exp(torch.complex(zero,f1_)))
        # print(torch.exp(torch.complex(zero,-f2_)))
        distance = torch.real(torch.exp(torch.complex(zero,f1_)) * torch.exp(torch.complex(zero,-f2_) ))
        # distance_vec = (-distance + 1)/2 + torch.abs(f2 - f1 +0.55) - torch.abs(f2 - f1 +0.5) + torch.abs(f2 - f1 +0.45) + torch.abs(f2 - f1-0.55) - torch.abs(f2 - f1-0.5) + torch.abs(f2 - f1-0.45) - torch.square(f1 - f2) - 1 #- torch.abs(f2) #torch.clamp(torch.abs(f2+0.55),-1,1) - torch.clamp(torch.abs(f2+0.45),-1,1)# + torch.clamp(f2, -1,1)  
        # distance_vec = (-distance + 1)/2 + 
        # distance_vec = (-distance + 1)/2 - 2*torch.abs(f1 + f2) + torch.abs(f1 + f2 +0.55) - torch.abs(f1 + f2 +0.5) + torch.abs(f1 + f2 +0.45) + torch.abs(f1 + f2-0.55) - torch.abs(f1 + f2-0.5) + torch.abs(f1 - f2-0.45) #- torch.square(f1 - f2) #- 2*f1  #- torch.abs(f2) #torch.clamp(torch.abs(f2+0.55),-1,1) - torch.clamp(torch.abs(f2+0.45),-1,1)# + torch.clamp(f2, -1,1)  
        distance_vec = (-distance + 1)/2
        condition1 = torch.logical_and(f2-f1 >= 0.5, f2-f1 <= 0.6)
        condition2 = torch.logical_and(f2-f1 >= -0.5, f2-f1 <= -0.4)
        distance_vec = torch.where(condition1  ,  -f1 - f2 , distance_vec)
        distance_vec = torch.where(condition2  , -f1 - f2 , distance_vec)

        # condition3 = torch.logical_and(f2-f1 >= 0.4, f2-f1 <= 0.5)
        # condition4 = torch.logical_and(f2-f1 >= -0.6, f2-f1 <= -0.5)
        # distance_vec = torch.where(condition3  , -2 * ( f1-f2) , distance_vec)
        # distance_vec = torch.where(condition4  , -2 * ( f1-f2) , distance_vec)
        # distance_vec+= torch.square(f1 + f2) #torch.abs(torch.square(f1 + f2))
        # print(distance_vec)

        # print(distance_vec)
        # loss_per_atom = 0.5 *  torch.sum((distance_vec)**2, dim=1) * used_sigmas_per_atom**2
        # loss_per_atom = torch.sum((distance_vec)**2, dim=1)
        return torch.sum(distance_vec)
##### same positions have value 1
##### -1 is the furthest away
# print(torch.real(torch.tensor([15.])))

# a = torch.tensor([0.1, 0.1, 0.1])
# b = torch.tensor([0.3, 0.3, 0.3])
# print("a", a)
# print("b", b)
# print("a, b \t", periodic_distance_3D_loss(a, b))


# a = torch.tensor([0.1, 0.5, 0.1])
# b = torch.tensor([0.9, 0.0, 0.3])
# print("a", a)
# print("b", b)
# print("a, b \t", periodic_distance_3D_loss(a, b))

# print("ind a, ind b \t", periodic_distance_3D_loss(torch.tensor([0.1]), torch.tensor([0.9])))
# print("ind a, ind b \t", periodic_distance_3D_loss(torch.tensor([0.5]), torch.tensor([0.0])))
# print("ind a, ind b \t", periodic_distance_3D_loss(torch.tensor([0.1]), torch.tensor([0.3])))

b_s = torch.tensor([i for i in np.linspace(-1.5,1.5, 500)], requires_grad=True).to(torch.float32)
a = torch.rand(1) *2 - 1
a = torch.tensor([0])

a_s = torch.repeat_interleave(a, 500)

distances = []
distances_grad = []
distances_modified = []
distances_grad_modified = []
distances_mse = []
distances_mse_grad = []

for a,b in zip (a_s, b_s):
    per = periodic_distance_3D_loss(a,b)
    distances.append(per.detach().numpy())
#  per.backward()
    distances_grad.append(torch.autograd.grad(per, b)[0].item())

    per_modified = mse_distance_modified(a,b)
    distances_modified.append(per_modified.detach().numpy())
    #  per.backward()
    distances_grad_modified.append(torch.autograd.grad(per_modified, b)[0].item())


    mse = mse_distance(a,b)
    distances_mse.append(mse.detach().numpy())
#  mse.backward()
    distances_mse_grad.append(torch.autograd.grad(mse, b)[0].item())
# distances = periodic_distance_3D_loss(a_s,b_s)
print(a_s.shape)
# print(distances.shape)
# print(distances.shape)
fig = plt.figure(figsize=(6,5),layout="constrained")
    # ax = fig.subplot_mosaic("""AAA
    #                         BCD""")
# ax = fig.subplot_mosaic("""AB""")
# ax = fig.subplot_mosaic("""A""")
ax = fig.subplot_mosaic("""B""")

# ax["A"].hlines(0, -2,2,colors='black')
# ax["B"].hlines(0, -2,2,colors='black')
        

# ax["A"].set_title("Loss")
# ax["B"].set_title("Gradient w.r.t the second coordinate")

# ax["A"].plot([i for i in np.linspace(-2,2, 500)],distances, label ='Periodic loss')
# ax["B"].plot([i for i in np.linspace(-2,2, 500)],distances_grad, label ='Periodic loss grad')

# ax["A"].plot([i for i in np.linspace(-2,2, 500)],distances_modified, label ='Periodic MSE')
# ax["B"].plot([i for i in np.linspace(-2,2, 500)],distances_grad_modified, label ='Periodic MSE grad')

# ax["A"].plot([i for i in np.linspace(-2,2, 500)],distances_mse, label ='MSE')
# ax["B"].plot([i for i in np.linspace(-2,2, 500)],distances_mse_grad, label ='MSE grad')
# # plt.xticks([-1, -0.5,0,0.5,1],["coord - 1", "coord - 0.5", "coord", "coord + 0.5", "coord + 1"])
# a_round = - round(float(a)*100)/100
# print(a_round)
# plt.xticks([a_round - 1, a_round - 0.5, a_round, a_round +0.5, a_round+1],[f"{a_round} - 1", f"{a_round} - 0.5", f"{a_round}", f"{a_round} + 0.5", f"{a_round} + 1"])
# ax["A"].legend(loc=1)
# ax["B"].legend(loc=1)
# plt.show()

a=0

a_round = 0


# ax["A"].hlines(0, -1.5,1.5,colors='black')
# # ax["A"].set_title("Loss")
# ax["A"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances, label ='$\mathcal{L}_{angular}$', linewidth = 3, alpha=0.7)
# ax["A"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances_modified, label ='$\mathcal{L}_{perMSE}$', linewidth = 3, alpha=0.7)
# ax["A"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances_mse, label ='$\mathcal{L}_{MSE}$', linewidth = 3, alpha=0.7)
# ax["A"].legend(loc=2, fontsize=16)

ax["B"].hlines(0, -1.5,1.5,colors='black')
# ax["B"].set_title("Gradient w.r.t the second coordinate")
ax["B"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances_grad, label ='$ \\nabla_{\hat{z}_{f}}\mathcal{L}_{angular}$', linewidth = 3, alpha=0.7)
ax["B"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances_grad_modified, label ='$\\nabla_{\hat{z}_{f}}\mathcal{L}_{perMSE}$', linewidth = 3, alpha=0.7)
ax["B"].plot([i for i in np.linspace(-1.5,1.5, 500)],distances_mse_grad, label ='$\\nabla_{\hat{z}_{f}}\mathcal{L}_{MSE}$', linewidth = 3, alpha=0.7)
ax["B"].legend(loc=2, fontsize=16)

# plt.xticks([-1, -0.5,0,0.5,1],["coord - 1", "coord - 0.5", "coord", "coord + 0.5", "coord + 1"])
# a_round = - round(float(a)*100)/100
# print(a_round)
# plt.xticks([a_round - 1, a_round - 0.5, a_round, a_round +0.5, a_round+1],[f"{a_round} - 1", f"{a_round} - 0.5", f"{a_round}", f"{a_round} + 0.5", f"{a_round} + 1"])

plt.xticks([a_round - 1.5, a_round - 1, a_round - 0.5, a_round, a_round +0.5, a_round+1, a_round +1.5],\
           ["$z_f$ - 1.5", "$z_f$ - 1", "$z_f$ - 0.5", "$z_f$", "$z_f$ + 0.5", "$z_f$ + 1","$z_f$ + 1.5"], \
            fontsize = 14)
plt.yticks(fontsize = 14)
plt.savefig('losses_grad_new.pdf')
plt.show()




# print("b, a \t", periodic_distance_3D(b, a))
# print("euc\t", torch.matmul(a-b,a-b))
# print('\n')
# print("a, a \t", periodic_distance_3D(a, a))
# print("euc a a\t", torch.matmul(a-a,a-a))
# print('\n\n')
# a = torch.tensor([0.1, 0.1, 0.1])
# b = torch.tensor([0.6, 0.3, 0.3])
# print("a, b \t", periodic_distance_3D(a, b))
# # print("b, a \t", periodic_distance_3D(b, a))
# print("euc\t", torch.matmul(a-b,a-b))

# a = torch.tensor([0.1, 0.1, 0.1])
# b = torch.tensor([0.6, 0.6, 0.6])
# print("a, b \t", periodic_distance_3D(a, b))
# print("euc\t", torch.matmul(a-b,a-b))

# a = torch.tensor([0.1, 0.1, 0.1])
# b = torch.tensor([-0.4, -0.4, -0.4])
# print("a, b \t", periodic_distance_3D(a, b))

# a = torch.tensor([0.6, 0.6, 0.6])
# b = torch.tensor([-0.4, -0.4, -0.4])
# print("a, b \t", periodic_distance_3D(a, b))

# print('\n\n\n')
# # a = torch.tensor([0.1, 0.1, 0.1])
# # b = torch.tensor([0.9, 0.9, 0.9])
# print("a, b \t", periodic_distance(a, b))
# # print("b, a \t", periodic_distance(b, a))

# # a = torch.tensor([1])
# # b = torch.tensor([1.5])
# # print("a, b \t", periodic_distance(a, b))
# # print("b, a \t", periodic_distance(b, a))


# # a = torch.tensor([0.])
# # b = torch.tensor([0.5])
# # print("a, b \t", periodic_distance(a, b))
# # print("b, a \t", periodic_distance(b, a))