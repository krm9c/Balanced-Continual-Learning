import torch
import numpy as np
from importlib import import_module
from .default import NormalNN
from .regularization import SI, L2, EWC, MAS
from dataloaders.wrapper import Storage
import copy 

class Naive_Rehearsal(NormalNN):

    def __init__(self, agent_config):
        super(Naive_Rehearsal, self).__init__(agent_config)
        self.task_count = 0
        self.memory_size = 1000
        self.task_memory = {}
        self.skip_memory_concatenation = False

    def learn_batch(self, train_loader, val_loader=None):
        # 1.Combine training set
        if self.skip_memory_concatenation:
            new_train_loader = train_loader
        else: # default
            dataset_list = []
            for storage in self.task_memory.values():
                dataset_list.append(storage)
            dataset_list *= max(len(train_loader.dataset)//self.memory_size,1)  # Let old data: new data = 1:1
            dataset_list.append(train_loader.dataset)
            dataset = torch.utils.data.ConcatDataset(dataset_list)
            new_train_loader = torch.utils.data.DataLoader(dataset,
                                                        batch_size=train_loader.batch_size,
                                                        shuffle=True,
                                                        num_workers=train_loader.num_workers)

        # 2.Update model as normal
        super(Naive_Rehearsal, self).learn_batch(new_train_loader, val_loader)

        # 3.Randomly decide the images to stay in the memory
        self.task_count += 1
        # (a) Decide the number of samples for being saved
        num_sample_per_task = self.memory_size // self.task_count
        num_sample_per_task = min(len(train_loader.dataset),num_sample_per_task)
        # (b) Reduce current exemplar set to reserve the space for the new dataset
        for storage in self.task_memory.values():
            storage.reduce(num_sample_per_task)
        # (c) Randomly choose some samples from new task and save them to the memory
        randind = torch.randperm(len(train_loader.dataset))[:num_sample_per_task]  # randomly sample some data
        self.task_memory[self.task_count] = Storage(train_loader.dataset, randind)


class Naive_Rehearsal_SI(Naive_Rehearsal, SI):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_SI, self).__init__(agent_config)


class Naive_Rehearsal_L2(Naive_Rehearsal, L2):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_L2, self).__init__(agent_config)


class Naive_Rehearsal_EWC(Naive_Rehearsal, EWC):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_EWC, self).__init__(agent_config)
        self.online_reg = True  # Online EWC


class Naive_Rehearsal_MAS(Naive_Rehearsal, MAS):

    def __init__(self, agent_config):
        super(Naive_Rehearsal_MAS, self).__init__(agent_config)


class GEM(Naive_Rehearsal):
    """
    @inproceedings{GradientEpisodicMemory,
        title={Gradient Episodic Memory for Continual Learning},
        author={Lopez-Paz, David and Ranzato, Marc'Aurelio},
        booktitle={NIPS},
        year={2017},
        url={https://arxiv.org/abs/1706.08840}
    }
    """

    def __init__(self, agent_config):
        super(GEM, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.quadprog = import_module('quadprog')
        self.task_mem_cache = {}

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def project2cone2(self, gradient, memories):
        """
            Solves the GEM dual QP described in the paper given a proposed
            gradient "gradient", and a memory of task gradients "memories".
            Overwrites "gradient" with the final projected update.

            input:  gradient, p-vector
            input:  memories, (t * p)-vector
            output: x, p-vector

            Modified from: https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py#L70
        """
        margin = self.config['reg_coef']
        memories_np = memories.cpu().contiguous().double().numpy()
        gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        #print(memories_np.shape, gradient_np.shape)
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose())
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        P = P + G * 0.001
        h = np.zeros(t) + margin
        v = self.quadprog.solve_qp(P, q, G, h)[0]
        x = np.dot(v, memories_np) + gradient_np
        new_grad = torch.Tensor(x).view(-1)
        if self.gpu:
            new_grad = new_grad.cuda()
        return new_grad

    def learn_batch(self, train_loader, val_loader=None):

        # Update model as normal
        super(GEM, self).learn_batch(train_loader, val_loader)

        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

    def update_model(self, inputs, targets, tasks):

        # compute gradient on previous tasks
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                self.zero_grad()
                # feed the data from memory and collect the gradients
                mem_out = self.forward(self.task_mem_cache[t]['data'])
                mem_loss = self.criterion(mem_out, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])
                mem_loss.backward()
                # Store the grads
                self.task_grads[t] = self.grad_to_vector()

        # now compute the grad on the current minibatch
        out = self.forward(inputs)
        loss = self.criterion(out, targets, tasks)
        self.optimizer.zero_grad()
        loss.backward()

        # check if gradient violates constraints
        if self.task_count > 0:
            current_grad_vec = self.grad_to_vector()
            mem_grad_vec = torch.stack(list(self.task_grads.values()))
            dotp = current_grad_vec * mem_grad_vec
            dotp = dotp.sum(dim=1)
            if (dotp < 0).sum() != 0:
                new_grad = self.project2cone2(current_grad_vec, mem_grad_vec)
                # copy gradients back
                self.vector_to_grad(new_grad)

        self.optimizer.step()
        return loss.detach(), out


# ###########################################################################
class DPMCL(Naive_Rehearsal):
    """
    "DPMCL, within this framework people"
    """
    def __init__(self, agent_config):
        super(DPMCL, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.task_mem_cache = {}
        self.k_range=agent_config['k_range']
        self.beta=agent_config['beta']
        self.epsilon=agent_config['epsilon']
        self.x_updates = agent_config['x_updates']
        self.theta_updates=agent_config['theta_updates']
        self.internal_batch_size=agent_config['internal_batch_size']
        self.lr_theta = agent_config['lr_theta']


    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def normalize_grad(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

    def learn_batch(self, train_loader, val_loader=None):
        # Update model as normal
        super(DPMCL, self).learn_batch(train_loader, val_loader)
        # print("I am bloody here in learn batch")
        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}

     ######################################################
    # This the funciton which needs to be optimized.
    ######################################################
    def update_model(self, inputs, targets, tasks):
        # compute gradient on previous tasks
        xx_list =[]
        yy_list =[]
        tt_list =()
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                ############################################
                ## Collect x_PN accross all the tasks
                ############################################
                xx = self.task_mem_cache[t]['data']
                yy = self.task_mem_cache[t]['target']
                tt = self.task_mem_cache[t]['task']

                # print("The tasks are", t, xx.shape, yy.shape)
                xx_list.append(xx)
                yy_list.append(yy)
                tt_list += tt
                # print(xx_list[t-1].shape, yy_list[t-1].shape)

            #print(len(xx_list), len(yy_list), len(tt_list))
            # xx_list = xx_list+inputs
            # yy_list = yy_list+targets
            # tt_list+=tasks
            xx_list = torch.cat(xx_list, dim = 0)
            yy_list = torch.cat(yy_list, dim = 0)
            # print(xx_list.shape, yy_list.shape, len(tt_list) )
            xx = torch.cat([xx_list,inputs], dim = 0)
            yy = torch.cat([yy_list,targets], dim = 0)
            
        for kappa in range(self.k_range):
            # print("I am in NASH")
            # now compute the grad on the current minibatch
            if self.task_count > 0:
                out = self.forward(inputs)
                Total_Loss =self.beta*self.criterion(out, targets, tasks)
                index = torch.tensor(np.random.randint(0, xx.shape[0], self.internal_batch_size))
                x_PN = xx[index]
                y_PN = yy[index]
                # print("The max target is", torch.max(y_PN) )
                tt_list = (tt_list+tasks)
                task_PN = []

                for i, element in enumerate(index[:]):
                    task_PN.append(tt_list[element])
                task_PN = tuple(task_PN)

                # # #########################################################
                # # PLAYER -- 1
                # # #########################################################
                # # ## The next function captures the uncertainty due to data
                # # ########################################################
                # # Player 1 Strategies
                # J_PN_x = self.criterion(self.forward(x_PN), y_PN, task_PN)
                # x_PN_temp = copy.deepcopy(x_PN)
                # x_PN_temp.requires_grad = True
                # adv_grad = 0
                # epsilon = self.epsilon
                # #########################################################################################
                # for epoch in range(self.x_updates): 
                #     x_PN_temp = x_PN_temp + epsilon*adv_grad 
                #     adv_grad = torch.autograd.grad(\
                #         self.criterion(self.forward(x_PN_temp), y_PN, task_PN),
                #         x_PN_temp)[0]
                #     # Normalize the gradient values.
                #     adv_grad = self.normalize_grad(\
                #         adv_grad, p=2,\
                #         dim=1, eps=1e-12)
                # #########################################################################################    
                # J_x = (self.criterion(self.forward(x_PN_temp), y_PN, task_PN)-J_PN_x)
                
                
                ###########################################################
                ### PLAYER--2
                # ########################################################
                J_P = self.criterion(self.forward(x_PN), y_PN, task_PN)
                ## ##############################################################
                ## ## Uncertainty due to the model
                ## ##############################################################
                # cop = copy.deepcopy(self.model)
                # opt_buffer = torch.optim.SGD(cop.parameters(),lr = self.lr_theta) 
                # J_PN_theta = self.criterion(self.forward(x_PN), y_PN, task_PN)
                # for i in range(self.theta_updates):
                #     opt_buffer.zero_grad()
                #     self.criterion(cop.forward(x_PN), y_PN, task_PN).backward(retain_graph=True)
                #     opt_buffer.step()
                # J_th = (self.criterion(cop.forward(x_PN), y_PN, task_PN) - J_PN_theta)

                # print("The loss after the data update", self.criterion(
                #    self.forward(x_PN_temp), y_PN, task_PN).item())
                # print("The four losses J_P, J_th, J_x", J_P.item(), J_th.item(), J_x.item())
                # Total_Loss += J_P+ 0.99*torch.norm(J_x+J_th)
                
                Total_Loss += J_P

            else:
                out = self.forward(inputs)
                Total_Loss = self.criterion(out, targets, tasks)


            # print( len(xx_list), len(yy_list), len(list(tt_list)) ) 
            # Total_Loss.backward()
            # x= input("Hello there")
            self.zero_grad()
            Total_Loss.backward()
            self.optimizer.step()
        return Total_Loss.detach(), out


        
#####################################################################
class NASH(Naive_Rehearsal):
    """
    "NASH people, Nash, the great John Nash"
    """
    def __init__(self, agent_config):
        super(NASH, self).__init__(agent_config)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.task_grads = {}
        self.task_mem_cache = {}
        self.k_range=agent_config['k_range']
        self.beta=agent_config['beta']
        self.epsilon=agent_config['epsilon']
        self.x_updates = agent_config['x_updates']
        self.theta_updates=agent_config['theta_updates']
        self.internal_batch_size=agent_config['internal_batch_size']
        self.lr_theta = agent_config['lr_theta']


        # self.k_range=3
        # self.beta=0.99
        # self.epsilon=0.1
        # self.x_updates = 2
        # self.theta_updates=5
        # self.internal_batch_size=128 
        # self.lr_theta = 0.001

    def grad_to_vector(self):
        vec = []
        for n,p in self.params.items():
            if p.grad is not None:
                vec.append(p.grad.view(-1))
            else:
                # Part of the network might has no grad, fill zero for those terms
                vec.append(p.data.clone().fill_(0).view(-1))
        return torch.cat(vec)

    def vector_to_grad(self, vec):
        # Overwrite current param.grad by slicing the values in vec (flatten grad)
        pointer = 0
        for n, p in self.params.items():
            # The length of the parameter
            num_param = p.numel()
            if p.grad is not None:
                # Slice the vector, reshape it, and replace the old data of the grad
                p.grad.copy_(vec[pointer:pointer + num_param].view_as(p))
                # Part of the network might has no grad, ignore those terms
            # Increment the pointer
            pointer += num_param

    def normalize_grad(self, input, p=2, dim=1, eps=1e-12):
        return input / input.norm(p, dim, True).clamp(min=eps).expand_as(input)

    ######################################################
    # This the funciton which needs to be optimized.
    ###################################################### 
    def update_model(self, inputs, targets, tasks):
        # print("I am in NASH", self.task_count)
        # compute gradient on previous tasks
        xx_list =[]
        yy_list =[]
        tt_list =()
        if self.task_count > 0:
            for t,mem in self.task_memory.items():
                ############################################
                ## Collect x_PN accross all the tasks
                ############################################
                xx = self.task_mem_cache[t]['data']
                yy = self.task_mem_cache[t]['target']
                tt = self.task_mem_cache[t]['task']

                # print("The tasks are", t, xx.shape, yy.shape)
                xx_list.append(xx)
                yy_list.append(yy)
                tt_list += tt
                # print(xx_list[t-1].shape, yy_list[t-1].shape)

            #print(len(xx_list), len(yy_list), len(tt_list))
            # xx_list = xx_list+inputs
            # yy_list = yy_list+targets
            # tt_list+=tasks
            xx_list = torch.cat(xx_list, dim = 0)
            yy_list = torch.cat(yy_list, dim = 0)
            # print(xx_list.shape, yy_list.shape, len(tt_list) )
            xx = torch.cat([xx_list,inputs], dim = 0)
            yy = torch.cat([yy_list,targets], dim = 0)
            
        for kappa in range(self.k_range):
            # print("I am in NASH")
            # now compute the grad on the current minibatch
            if self.task_count > 0:
                out = self.forward(inputs)
                Total_Loss =self.beta*self.criterion(out, targets, tasks)
                index = torch.tensor(np.random.randint(0, xx.shape[0], self.internal_batch_size))
                x_PN = xx[index]
                y_PN = yy[index]
                # print("The max target is", torch.max(y_PN) )
                tt_list = (tt_list+tasks)
                task_PN = []
                for i, element in enumerate(index[:]):
                    task_PN.append(tt_list[element])
                task_PN = tuple(task_PN)

                # #########################################################
                # PLAYER -- 1
                # #########################################################
                # ## The next function captures the uncertainty due to data
                # ########################################################
                # Player 1 Strategies
                J_PN_x = self.criterion(self.forward(x_PN), y_PN, task_PN)
                x_PN_temp = copy.deepcopy(x_PN)
                x_PN_temp.requires_grad = True
                adv_grad = 0
                epsilon = self.epsilon
                #########################################################################################
                for epoch in range(self.x_updates): 
                    x_PN_temp = x_PN_temp + epsilon*adv_grad 
                    adv_grad = torch.autograd.grad(\
                        self.criterion(self.forward(x_PN_temp), y_PN, task_PN),
                        x_PN_temp)[0]
                    # Normalize the gradient values.
                    adv_grad = self.normalize_grad(\
                        adv_grad, p=2,\
                        dim=1, eps=1e-12)
                #########################################################################################    
                J_x = (self.criterion(self.forward(x_PN_temp), y_PN, task_PN)-J_PN_x)
                # #########################################################
                # ## PLAYER--2
                # ########################################################
                J_P = self.criterion(self.forward(x_PN), y_PN, task_PN)
                # ##############################################################
                # ## Uncertainty due to the model
                # ##############################################################
                cop = copy.deepcopy(self.model)
                opt_buffer = torch.optim.SGD(cop.parameters(),lr = self.lr_theta) 
                J_PN_theta = self.criterion(self.forward(x_PN), y_PN, task_PN)
                for i in range(self.theta_updates):
                    opt_buffer.zero_grad()
                    self.criterion(cop.forward(x_PN), y_PN, task_PN).backward(retain_graph=True)
                    opt_buffer.step()
                J_th = (self.criterion(cop.forward(x_PN), y_PN, task_PN) - J_PN_theta)

                # print("The loss after the data update", self.criterion(
                #    self.forward(x_PN_temp), y_PN, task_PN).item())
                # print("The four losses J_P, J_th, J_x", J_P.item(), J_th.item(), J_x.item())
                # Total_Loss += J_P+ 0.99*torch.norm(J_x+J_th)
                Total_Loss += J_P+J_th+J_x

            else:
                out = self.forward(inputs)
                Total_Loss = self.criterion(out, targets, tasks)


            # print( len(xx_list), len(yy_list), len(list(tt_list)) ) 
            # Total_Loss.backward()
            # x= input("Hello there")
            self.zero_grad()
            Total_Loss.backward()
            self.optimizer.step()
        return Total_Loss.detach(), out



    def learn_batch(self, train_loader, val_loader=None):
        # Update model as normal
        super(NASH, self).learn_batch(train_loader, val_loader)
        # print("I am bloody here in learn batch")
        # Cache the data for faster processing
        for t, mem in self.task_memory.items():
            # print("I am loading the data")
            # Concatenate all data in each task
            mem_loader = torch.utils.data.DataLoader(mem,
                                                     batch_size=len(mem),
                                                     shuffle=False,
                                                     num_workers=2)
            assert len(mem_loader)==1,'The length of mem_loader should be 1'
            for i, (mem_input, mem_target, mem_task) in enumerate(mem_loader):
                if self.gpu:
                    mem_input = mem_input.cuda()
                    mem_target = mem_target.cuda()
            self.task_mem_cache[t] = {'data':mem_input,'target':mem_target,'task':mem_task}
    
     






    # # ######################################################
    # # # This the funciton which needs to be optimized.
    # # ###################################################### 
    # # def update_model(self, inputs, targets, tasks):


    # #     # now compute the grad on the current minibatch
    # #     out = self.forward(inputs)
    # #     Total_Loss= self.criterion(out, targets, tasks)

    # #     # compute gradient on previous tasks
    # #     xx_list =[]
    # #     yy_list =[]
    # #     tt_list =()
        
    # #     if self.task_count > 0:
    # #         # print(inputs, targets, self.task_mem_cache['1'])
    # #         # x = input()
    # #         J_P=0.0
    # #         for t,mem in self.task_memory.items():
    # #             # print("t", t, mem)
    # #             # feed the data from memory and collect the gradients
    # #             mem_out_JP = self.forward(self.task_mem_cache[t]['data'])
    # #             # print(mem_out, self.task_mem_cache[t]['target'],\
    # #             #      self.task_mem_cache[t]['task'])
    # #             J_P+=self.criterion(mem_out_JP, self.task_mem_cache[t]['target'], self.task_mem_cache[t]['task'])

    # #         ############################################
    # #         ## The next function captures the uncertainty due to the model
    # #         ############################################
    # #         cop = copy.deepcopy(self.model)
    # #         opt_buffer = torch.optim.SGD(cop.parameters(),lr = 0.001) 
    # #         for i in range(2):
    # #             opt_buffer.zero_grad()
    # #             JPN_out = cop.forward(inputs)
    # #             J_PN_1= self.criterion(out, targets, tasks)
    # #             JJ=0.0
    # #             for t,mem in self.task_memory.items():
    # #                 # print("t", t, mem)
    # #                 # feed the data from memory and collect the gradients
    # #                 mem_out_JPN1 = cop.forward(self.task_mem_cache[t]['data'])
    # #                 # print(mem_out, self.task_mem_cache[t]['target'],\
    # #                 #      self.task_mem_cache[t]['task'])
    # #                 JJ+=self.criterion(mem_out_JPN1, self.task_mem_cache[t]['target'],\
    # #                      self.task_mem_cache[t]['task'])
    # #             J_PN_1+=JJ
    # #             J_PN_1.backward(retain_graph=True)
    # #             opt_buffer.step()
                
    # #         ############################################
    # #         JPN_out = cop.forward(inputs)
    # #         J_PN_1= self.criterion(out, targets, tasks)
    # #         JJ=0.0
    # #         for t,mem in self.task_memory.items():
    # #             # print("t", t, mem)
    # #             # feed the data from memory and collect the gradients
    # #             mem_out_JPN1 = cop.forward(self.task_mem_cache[t]['data'])
    # #             # print(mem_out, self.task_mem_cache[t]['target'],\
    # #             #      self.task_mem_cache[t]['task'])
    # #             JJ+=self.criterion(mem_out_JPN1, self.task_mem_cache[t]['target'],\
    # #                     self.task_mem_cache[t]['task'])
    # #         J_PN_1+=JJ
    # #         ############################################
    # #         JJP_out = self.forward(inputs)
    # #         J_PN=self.criterion(out, targets, tasks)
    # #         JJP=0.0
    # #         for t,mem in self.task_memory.items():
    # #             # print("t", t, mem)
    # #             # feed the data from memory and collect the gradients
    # #             mem_out_JPN = self.forward(self.task_mem_cache[t]['data'])
    # #             # print(mem_out, self.task_mem_cache[t]['target'],\
    # #             #      self.task_mem_cache[t]['task'])
    # #             JJP+=self.criterion(mem_out_JPN, self.task_mem_cache[t]['target'],\
    # #                     self.task_mem_cache[t]['task'])
    # #         J_PN+=JJP



    # #         ############################################
    # #         ## The next function captures the uncertainty due to the model
    # #         ############################################
    # #         # Create my data sample
    # #         # print(inputs.shape, targets.shape, len(tasks))

    # #         for t,mem in self.task_memory.items():
    # #             xx = self.task_mem_cache[t]['data']
    # #             yy = self.task_mem_cache[t]['target']
    # #             tt = self.task_mem_cache[t]['task']
    # #             # print(xx.shape, yy.shape)
    # #             xx_list.append(xx)
    # #             yy_list.append(yy)
    # #             tt_list += tt


    # #         # print(len(xx_list), len(yy_list) )
    # #         # xx_list = xx_list+inputs
    # #         # yy_list = yy_list+targets
    # #         # tt_list+=tasks
    # #         xx_list = torch.stack(xx_list)[0]
    # #         yy_list = torch.stack(yy_list)[0]

    # #         # print(xx.shape, yy.shape, len(list(tt_list)) )
    # #         x_PN = torch.cat([xx_list,inputs], dim = 0)
    # #         y_PN = torch.cat([yy_list,targets], dim = 0)
    # #         index = np.random.randint(0, x_PN.shape[0], 128)


    # #         x_PN = x_PN[index,:, :, :]
    # #         y_PN = y_PN[index]
    # #         tt_list = (tt_list+tasks)
    # #         task_PN = []           
    # #         for i,element in enumerate(index[:]):
    # #             task_PN.append(tt_list[element])

    # #         task_PN =tuple(task_PN)
    # #         # print(x_PN.shape, y_PN.shape, len(list(task_PN)) ) 
    # #         # print(x_PN, y_PN, task_PN)
    # #         mem_out_xPN = self.forward(x_PN)
    # #         Test_criterion=self.criterion(mem_out_xPN, y_PN, task_PN)
            


    # #         # Player 1 Strategies
    # #         x_PN_temp = copy.deepcopy(x_PN)
    # #         x_PN_temp.requires_grad = True
    # #         adv_grad = 0
    # #         epsilon = 0.01
    # #         #########################################################################################
    # #         for epoch in range(2): 
    # #             x_PN_temp = x_PN_temp + epsilon*adv_grad 
    # #             mem_out_xPN_1 = self.forward(x_PN_temp)
    # #             Test_criterion_PN1=self.criterion(mem_out_xPN_1, y_PN, task_PN)
    # #             adv_grad = torch.autograd.grad(\
    # #                 Test_criterion_PN1,\
    # #                 x_PN_temp)[0]

    # #             # Normalize the gradient values.
    # #             adv_grad = self.normalize_grad(\
    # #                 adv_grad, p=2,\
    # #                 dim=1, eps=1e-12)
            
    # #         #########################################################################################    
    # #         mem_out_xPN_1 = self.forward(x_PN_temp)
    # #         Test_criterion_1 =self.criterion(\
    # #             mem_out_xPN_1, y_PN, task_PN)


    # #         Total_Loss += J_P + 0.99*(J_PN_1-J_PN) \
    # #         +0.99*(Test_criterion_1 - Test_criterion)
    # #     # print( len(xx_list), len(yy_list), len(list(tt_list)) ) 
    # #     # Total_Loss.backward()
    # #     self.zero_grad()
    # #     Total_Loss.backward()
    # #     self.optimizer.step()
    # #     return Total_Loss.detach(), out


    # ######################################################
    # # This the funciton which needs to be optimized.
    # ###################################################### 
    # def update_model(self, inputs, targets, tasks):
    #     # compute gradient on previous tasks
    #     xx_list =[]
    #     yy_list =[]
    #     tt_list =()
        
    #     if self.task_count > 0:
    #         for t,mem in self.task_memory.items():
    #             ############################################
    #             ## Collect x_PN accross all the tasks
    #             ############################################
    #             xx = self.task_mem_cache[t]['data']
    #             yy = self.task_mem_cache[t]['target']
    #             tt = self.task_mem_cache[t]['task']

    #             # print("The tasks are", t, xx.shape, yy.shape)
    #             xx_list.append(xx)
    #             yy_list.append(yy)
    #             tt_list += tt
    #             # print(xx_list[t-1].shape, yy_list[t-1].shape)

    #         # print(len(xx_list), len(yy_list), len(tt_list))
    #         # xx_list = xx_list+inputs
    #         # yy_list = yy_list+targets
    #         # tt_list+=tasks
    #         xx_list = torch.cat(xx_list, dim = 0)
    #         yy_list = torch.cat(yy_list, dim = 0)
    #         # print(xx_list.shape, yy_list.shape, len(tt_list) )
    #         xx = torch.cat([xx_list,inputs], dim = 0)
    #         yy = torch.cat([yy_list,targets], dim = 0)
            
    #     for kappa in range(10):
    #         # print("I am in NASH")
    #         # now compute the grad on the current minibatch
    #         out = self.forward(inputs)
    #         Total_Loss= self.criterion(out, targets, tasks)

    #         if self.task_count > 0:
    #             index = torch.tensor(np.random.randint(0, xx.shape[0], 128))
    #             x_PN = xx[index]
    #             y_PN = yy[index]

    #             # print("The max target is", torch.max(y_PN) )
    #             tt_list = (tt_list+tasks)
    #             task_PN = []           
    #             for i,element in enumerate(index[:]):
    #                 task_PN.append(tt_list[element])
    #             task_PN =tuple(task_PN)
    #             J_P = self.criterion(self.forward(x_PN), y_PN, task_PN)

                
    #             # ##############################################################
    #             # ## The next function captures the uncertainty due to the model
    #             # ##############################################################
    #             cop = copy.deepcopy(self.model)
    #             opt_buffer = torch.optim.SGD(cop.parameters(),lr = 0.001) 
    #             J_PN_theta = self.criterion(self.forward(x_PN), y_PN, task_PN)
    #             for i in range(5):
    #                 opt_buffer.zero_grad()
    #                 self.criterion(cop.forward(x_PN), y_PN, task_PN).backward(retain_graph=True)
    #                 opt_buffer.step()
    #             J_th = 0.99*(self.criterion(cop.forward(x_PN), y_PN, task_PN) - J_PN_theta)
                
                
    #             # #########################################################
    #             # ## The next function captures the uncertainty due to data
    #             # ########################################################
    #             # Player 1 Strategies
    #             J_PN_x = self.criterion(self.forward(x_PN), y_PN, task_PN)
    #             x_PN_temp = copy.deepcopy(x_PN)
    #             x_PN_temp.requires_grad = True
    #             adv_grad = 0
    #             epsilon = 0.01
    #             #########################################################################################
    #             for epoch in range(5): 
    #                 x_PN_temp = x_PN_temp + epsilon*adv_grad 
    #                 adv_grad = torch.autograd.grad(\
    #                     self.criterion(self.forward(x_PN_temp), y_PN, task_PN),
    #                     x_PN_temp)[0]

    #                 # Normalize the gradient values.
    #                 adv_grad = self.normalize_grad(\
    #                     adv_grad, p=2,\
    #                     dim=1, eps=1e-12)
                
    #         #########################################################################################    
    #             J_x = 0.99*(self.criterion(self.forward(x_PN_temp), y_PN, task_PN)-J_PN_x)
    #             Total_Loss+=J_P+J_th+J_x
    #         # print( len(xx_list), len(yy_list), len(list(tt_list)) ) 
    #         # Total_Loss.backward()
    #         self.zero_grad()
    #         Total_Loss.backward()
    #         self.optimizer.step()

    #     return Total_Loss.detach(), out


