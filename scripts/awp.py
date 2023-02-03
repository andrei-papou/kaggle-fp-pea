import torch
from torch.optim import Optimizer


class Loss:

    def __call__(self, logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()


class AWP:

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            loss_func: Loss,
            adv_param: str = 'weight',
            adv_lr: float = 1.0,
            adv_eps: float = 0.2,
            start_epoch: int = 0,
            accumultion: int = 1):
        self._model = model
        self._optimizer = optimizer
        self._loss_func = loss_func
        self._adv_param = adv_param
        self._adv_lr = adv_lr
        self._adv_eps = adv_eps
        self._start_epoch = start_epoch
        self._accumultion = accumultion

        self._backup = {}
        self._backup_eps = {}

    def attack_backward(self, inputs, epoch, device: str):
        if (self._adv_lr == 0.0) or (epoch < self._start_epoch):
            return None

        self._save()
        self._attack_step()
        self._optimizer.zero_grad()
        for model_input in inputs:
            for k, v in model_input.items():
                model_input[k] = v.to(device)
            out = self._model(model_input)
            adv_loss = self._loss_func(out, model_input['label'])
            adv_loss = adv_loss/self._accumultion
            adv_loss.backward()
        # del model_input, inputs, adv_loss, out
        torch.cuda.empty_cache()
        self._restore()

    def _attack_step(self):
        e = 1e-6
        for name, param in self._model.named_parameters():
            if param.requires_grad and param.grad is not None and self._adv_param in name:
                norm1 = torch.norm(param.grad)
                norm2 = torch.norm(param.data.detach())
                if norm1 != 0 and not torch.isnan(norm1):
                    r_at = self._adv_lr * param.grad / (norm1 + e) * (norm2 + e)
                    param.data.add_(r_at)
                    param.data = torch.min(
                        torch.max(param.data, self._backup_eps[name][0]), self._backup_eps[name][1]
                    )

    def _save(self):
        for name, param in self._model.named_parameters():
            if param.requires_grad and param.grad is not None and self._adv_param in name:
                if name not in self._backup:
                    self._backup[name] = param.data.clone()
                    grad_eps = self._adv_eps * param.abs().detach()
                    self._backup_eps[name] = (
                        self._backup[name] - grad_eps,
                        self._backup[name] + grad_eps,
                    )

    def _restore(self,):
        for name, param in self._model.named_parameters():
            if name in self._backup:
                param.data = self._backup[name]
        self._backup = {}
        self._backup_eps = {}


class AWPBuilder:

    def __init__(
            self,
            adv_param: str = 'weight',
            adv_lr: float = 1.0,
            adv_eps: float = 0.2,
            start_epoch: int = 0,
            accumultion: int = 1):
        self._adv_param = adv_param
        self._adv_lr = adv_lr
        self._adv_eps = adv_eps
        self._start_epoch = start_epoch
        self._accumultion = accumultion

    def build(
            self,
            model: torch.nn.Module,
            optimizer: Optimizer,
            loss_func: Loss) -> AWP:
        return AWP(
            model=model,
            optimizer=optimizer,
            loss_func=loss_func,
            adv_param=self._adv_param,
            adv_lr=self._adv_lr,
            adv_eps=self._adv_eps,
            start_epoch=self._start_epoch,
            accumultion=self._accumultion)
