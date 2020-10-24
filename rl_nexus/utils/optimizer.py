import time
import math
import torch
import torch.autograd as autograd

class OptimisticAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(OptimisticAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(OptimisticAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # self.state[p]['step'] = 0
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    self.state[p]['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # import pdb; pdb.set_trace()
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # import pdb;pdb.set_trace()
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    # import pdb; pdb.set_trace()
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['num'] = [torch.zeros_like(p, memory_format=torch.preserve_format), torch.zeros_like(p, memory_format=torch.preserve_format)]
                    state['denom'] = [torch.zeros_like(p, memory_format=torch.preserve_format), torch.zeros_like(p, memory_format=torch.preserve_format)+group['eps']]
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state['num'][0] = state['num'][1]
                state['denom'][0] = state['denom'][1]
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # print(bias_correction2)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                num = exp_avg / bias_correction1
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq/ bias_correction2, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt()).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                
                state['num'][1] = num.clone()
                
                state['denom'][1] = denom.clone()
                # if state['step'] == 1:
                #     grad_term = num/denom
                # else:
                grad_term = 2*num/denom -  state['num'][0]/state['denom'][0]
                # p = p - group['lr']*grad_term
                p.add_(- group['lr']*grad_term)
                # import pdb;pdb.set_trace()
                # step_size = group['lr'] / bias_correction1
                # print(bias_correction2)
                # import pdb;pdb.set_trace()

                # p.addcdiv_(exp_avg, denom, value=-step_size)
        return grad_term
        # return exp_avg / denom
        # return loss

# from .cgd_utils import zero_grad, general_conjugate_gradient, Hvp_vec, gd_solver

def conjugate_gradient(grad_x, grad_y,
                       x_params, y_params,
                       b, x=None, nsteps=10,
                       tol=1e-10, atol=1e-16,
                       lr_x=1.0, lr_y=1.0,
                       device=torch.device('cpu')):
    """
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b: vec
    :param nsteps: max number of steps
    :param residual_tol:
    :return: A ** -1 * b
    h_1 = D_yx * p
    h_2 = D_xy * D_yx * p
    A = I + lr_x * D_xy * lr_y * D_yx
    """
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=x, retain_graph=True).detach_().mul(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).detach_().mul(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    if rdotr < residual_tol or rdotr < atol:
        return x, 1

    for i in range(nsteps):
        # To compute Avp
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=p, retain_graph=True).detach_().mul(lr_y)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h_1, retain_graph=True).detach_().mul(lr_x)
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)

        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    if i > 99:
        warnings.warn('CG iter num: %d' % (i + 1))
    return x, i + 1


def Hvp_vec(grad_vec, params, vec, retain_graph=False):
    '''
    Parameters:
        - grad_vec: Tensor of which the Hessian vector product will be computed
        - params: list of params, w.r.t which the Hessian will be computed
        - vec: The "vector" in Hessian vector product
    return: Hessian vector product
    '''
    if torch.isnan(grad_vec).any():
        raise ValueError('Gradvec nan')
    if torch.isnan(vec).any():
        raise ValueError('vector nan')
        # zero padding for None
    grad_grad = autograd.grad(grad_vec, params, grad_outputs=vec, retain_graph=retain_graph,
                              allow_unused=True)
    grad_list = []
    for i, p in enumerate(params):
        if grad_grad[i] is None:
            grad_list.append(torch.zeros_like(p).view(-1))
        else:
            grad_list.append(grad_grad[i].contiguous().view(-1))
    hvp = torch.cat(grad_list)
    if torch.isnan(hvp).any():
        raise ValueError('hvp Nan')
    return hvp


def general_conjugate_gradient(grad_x, grad_y,
                               x_params, y_params, b,
                               lr_x, lr_y, x=None, nsteps=None,
                               tol=1e-12, atol=1e-20,
                               device=torch.device('cpu')):
    '''
    :param grad_x:
    :param grad_y:
    :param x_params:
    :param y_params:
    :param b:
    :param lr_x:
    :param lr_y:
    :param x:
    :param nsteps:
    :param residual_tol:
    :param device:
    :return: (I + sqrt(lr_x) * D_xy * lr_y * D_yx * sqrt(lr_x)) ** -1 * b
    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
        r = b.clone().detach()
    else:
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params,
                     vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params,
                     vec=h1, retain_graph=True).mul_(lr_x)
        Avx = x + h2
        r = b.clone().detach() - Avx

    if nsteps is None:
        nsteps = b.shape[0]

    if grad_x.shape != b.shape:
        raise RuntimeError('CG: hessian vector product shape mismatch')
    p = r.clone().detach()
    rdotr = torch.dot(r, r)
    residual_tol = tol * rdotr
    if rdotr < residual_tol or rdotr < atol:
        return x, 1
    for i in range(nsteps):
        # To compute Avp
        # h_1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * p, retain_graph=True)
        h_1 = Hvp_vec(grad_vec=grad_x, params=y_params,
                      vec=lr_x * p, retain_graph=True).mul_(lr_y)
        # h_1.mul_(lr_y)
        # lr_y * D_yx * b
        # h_2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=lr_y * h_1, retain_graph=True)
        h_2 = Hvp_vec(grad_vec=grad_y, params=x_params,
                      vec=h_1, retain_graph=True).mul_(lr_x)
        # h_2.mul_(lr_x)
        # lr_x * D_xy * lr_y * D_yx * b
        Avp_ = p + h_2

        alpha = rdotr / torch.dot(p, Avp_)
        x.data.add_(alpha * p)
        r.data.add_(- alpha * Avp_)
        new_rdotr = torch.dot(r, r)
        beta = new_rdotr / rdotr
        p = r + beta * p
        rdotr = new_rdotr
        if rdotr < residual_tol or rdotr < atol:
            break
    return x, i + 1


def gd_solver(grad_x, grad_y,
              x_params, y_params,
              b, x=None, iter_num=1,
              lr_x=1.0, lr_y=1.0,
              rho=0.9, beta=0.1,
              device=torch.device('cpu')):
    '''
    slove inversion by gradient descent
    '''
    lr_x = lr_x.sqrt()
    if x is None:
        x = torch.zeros(b.shape[0], device=device)
    for i in range(iter_num):
        h1 = Hvp_vec(grad_vec=grad_x, params=y_params, vec=lr_x * x, retain_graph=True).mul_(lr_y)
        h2 = Hvp_vec(grad_vec=grad_y, params=x_params, vec=h1, retain_graph=True).mul_(lr_x)
        delta_x = x + h2 - b
        x = rho * x - beta * delta_x
    return x, 1


def zero_grad(params):
    for p in params:
        if p.grad is not None:
            p.grad.detach()
            p.grad.zero_()

class ACGD(object):
    def __init__(self, max_params, min_params,
                 lr_max=1e-3, lr_min=1e-3,
                 eps=1e-5, beta=0.99,
                 tol=1e-12, atol=1e-20,
                 device=torch.device('cpu'),
                 solve_x=False, collect_info=True,
                 solver='cg'):
        self.max_params = list(max_params)
        self.min_params = list(min_params)
        self.state = {'lr_max': lr_max, 'lr_min': lr_min,
                      'eps': eps, 'solve_x': solve_x,
                      'tol': tol, 'atol': atol,
                      'beta': beta, 'step': 0,
                      'old_max': None, 'old_min': None,  # start point of CG
                      'sq_exp_avg_max': None, 'sq_exp_avg_min': None}  # save last update
        self.info = {'grad_x': None, 'grad_y': None,
                     'hvp_x': None, 'hvp_y': None,
                     'cg_x': None, 'cg_y': None,
                     'time': 0, 'iter_num': 0}
        self.device = device
        self.collect_info = collect_info
        self.solver= solver

    def zero_grad(self):
        zero_grad(self.max_params)
        zero_grad(self.min_params)

    def get_info(self):
        if self.info['grad_x'] is None:
            print('Warning! No update information stored. Set collect_info=True before call this method')
        return self.info

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)
        print('Load state: {}'.format(state_dict))

    def set_lr(self, lr_max, lr_min):
        self.state.update({'lr_max': lr_max, 'lr_min': lr_min})
        print('Maximizing side learning rate: {:.4f}\n '
              'Minimizing side learning rate: {:.4f}'.format(lr_max, lr_min))

    def step(self, loss):
        lr_max = self.state['lr_max']
        lr_min = self.state['lr_min']
        beta = self.state['beta']
        eps = self.state['eps']
        tol = self.state['tol']
        atol = self.state['atol']
        time_step = self.state['step'] + 1
        self.state['step'] = time_step

        grad_x = autograd.grad(loss, self.max_params, create_graph=True, retain_graph=True)
        grad_x_vec = torch.cat([g.contiguous().view(-1) for g in grad_x])
        grad_y = autograd.grad(loss, self.min_params, create_graph=True, retain_graph=True)
        grad_y_vec = torch.cat([g.contiguous().view(-1) for g in grad_y])
        grad_x_vec_d = grad_x_vec.clone().detach()
        grad_y_vec_d = grad_y_vec.clone().detach()

        sq_avg_x = self.state['sq_exp_avg_max']
        sq_avg_y = self.state['sq_exp_avg_min']
        sq_avg_x = torch.zeros_like(grad_x_vec_d, requires_grad=False) if sq_avg_x is None else sq_avg_x
        sq_avg_y = torch.zeros_like(grad_y_vec_d, requires_grad=False) if sq_avg_y is None else sq_avg_y
        sq_avg_x.mul_(beta).addcmul_(1 - beta, grad_x_vec_d, grad_x_vec_d)
        sq_avg_y.mul_(beta).addcmul_(1 - beta, grad_y_vec_d, grad_y_vec_d)

        bias_correction = 1 - beta ** time_step
        lr_max = math.sqrt(bias_correction) * lr_max / sq_avg_x.sqrt().add(eps)
        lr_min = math.sqrt(bias_correction) * lr_min / sq_avg_y.sqrt().add(eps)

        scaled_grad_x = torch.mul(lr_max, grad_x_vec_d)
        scaled_grad_y = torch.mul(lr_min, grad_y_vec_d)
        hvp_x_vec = Hvp_vec(grad_y_vec, self.max_params, scaled_grad_y,
                            retain_graph=True)  # h_xy * d_y
        hvp_y_vec = Hvp_vec(grad_x_vec, self.min_params, scaled_grad_x,
                            retain_graph=True)  # h_yx * d_x
        p_x = torch.add(grad_x_vec_d, - hvp_x_vec)
        p_y = torch.add(grad_y_vec_d, hvp_y_vec)
        if self.collect_info:
            norm_px = torch.norm(hvp_x_vec, p=2).item()
            norm_py = torch.norm(hvp_y_vec, p=2).item()
            timer = time.time()

        if self.state['solve_x']:
            p_y.mul_(lr_min.sqrt())
            if self.solver == 'cg':
                cg_y, iter_num = general_conjugate_gradient(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                                            x_params=self.min_params, y_params=self.max_params,
                                                            b=p_y, x=self.state['old_min'],
                                                            tol=tol, atol=atol,
                                                            lr_x=lr_min, lr_y=lr_max, device=self.device)
            elif self.solver == 'gd':
                cg_y, iter_num = gd_solver(grad_x=grad_y_vec, grad_y=grad_x_vec,
                                           x_params=self.min_params, y_params=self.max_params,
                                           b=p_y, x=self.state['old_min'],
                                           lr_x=lr_min, lr_y=lr_max, device=self.device)
            old_min = cg_y.detach_()
            min_update = cg_y.mul(- lr_min.sqrt())
            hcg = Hvp_vec(grad_y_vec, self.max_params, min_update).detach_()
            hcg.add_(grad_x_vec_d)
            max_update = hcg.mul(lr_max)
            old_max = hcg.mul(lr_max.sqrt())
        else:
            p_x.mul_(lr_max.sqrt())
            if self.solver == 'cg':
                cg_x, iter_num = general_conjugate_gradient(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                                            x_params=self.max_params, y_params=self.min_params,
                                                            b=p_x, x=self.state['old_max'],
                                                            tol=tol, atol=atol,
                                                            lr_x=lr_max, lr_y=lr_min, device=self.device)
            elif self.solver == 'gd':
                cg_x, iter_num = gd_solver(grad_x=grad_x_vec, grad_y=grad_y_vec,
                                           x_params=self.max_params, y_params=self.min_params,
                                           b=p_x, x=self.state['old_max'],
                                           lr_x=lr_max, lr_y=lr_min, device=self.device)
            old_max = cg_x.detach_()
            max_update = cg_x.mul(lr_max.sqrt())
            hcg = Hvp_vec(grad_x_vec, self.min_params, max_update).detach_()
            hcg.add_(grad_y_vec_d)
            min_update = hcg.mul(- lr_min)
            old_min = hcg.mul(lr_min.sqrt())
        self.state.update({'old_max': old_max, 'old_min': old_min,
                           'sq_exp_avg_max': sq_avg_x, 'sq_exp_avg_min': sq_avg_y})

        if self.collect_info:
            timer = time.time() - timer
            self.info.update({'time': timer, 'iter_num': iter_num,
                              'hvp_x': norm_px, 'hvp_y': norm_py})

        index = 0
        for p in self.max_params:
            p.data.add_(max_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == max_update.numel(), 'Maximizer CG size mismatch'

        index = 0
        for p in self.min_params:
            p.data.add_(min_update[index: index + p.numel()].reshape(p.shape))
            index += p.numel()
        assert index == min_update.numel(), 'Minimizer CG size mismatch'

        if self.collect_info:
            norm_gx = torch.norm(grad_x_vec, p=2).item()
            norm_gy = torch.norm(grad_y_vec, p=2).item()
            norm_cgx = torch.norm(max_update, p=2).item()
            norm_cgy = torch.norm(min_update, p=2).item()
            self.info.update({'grad_x': norm_gx, 'grad_y': norm_gy,
                              'cg_x': norm_cgx, 'cg_y': norm_cgy})
        self.state['solve_x'] = False if self.state['solve_x'] else True


class ModifiedAdam(torch.optim.Optimizer):
    r"""Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        # if not 0.0 <= lr:
            # raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(ModifiedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ModifiedAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def reset(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # self.state[p]['step'] = 0
                self.state[p]['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                self.state[p]['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                if amsgrad:
                    # Maintains max of all exp. moving avg. of sq. grad. values
                    self.state[p]['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # import pdb; pdb.set_trace()
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]
                # import pdb;pdb.set_trace()
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                # print(bias_correction2)

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq/ bias_correction2, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt()).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1
                # print(bias_correction2)
                # import pdb;pdb.set_trace()

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return exp_avg / denom
        # return loss
