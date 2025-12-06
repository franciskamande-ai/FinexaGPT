import torch
import torch.nn as nn
import math
from omegaconf import DictConfig
import tqdm
# TODO:
'''
There is an error here:
The train_epoch function tries to enumerate a tuple
while our dataset class return's a dictionary ot input/output_ids  # SOLVED
Fix: access data n the epoch via batch['input_ids']
WILL DO IT LATER
'''

'''
Implementing tqdm to show progress could be fun 
and maybe beneficial to the one  training        # SOLVED
'''

class Trainer(nn.Module):
    def __init__(self, model, cfg: DictConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        self.num_training_steps = cfg.training.num_training_steps
        self.num_warmup_steps = cfg.training.num_warmup_steps
        self.min_lr = cfg.training.min_lr
        self.gradient_clip = cfg.training.gradient_clip
        
        self.optimizer = self._configure_optimizer()
        self.scheduler = self._learning_rate_scheduling()
        
        self.current_step = 0

    def _learning_rate_scheduling(self):
        def lr_lambda(current_step):
            if current_step < self.num_warmup_steps:
                return float(current_step) / float(max(1, self.num_warmup_steps))
            
            progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
            cos_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return max(self.min_lr, cos_decay)
        
        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def _configure_optimizer(self):
        weight_decay = self.cfg.training.weight_decay
        learning_rate = self.cfg.training.learning_rate
        
        param_dict = {name: param for name, param in self.model.named_parameters()}
        param_dict = {name: param for name, param in param_dict.items() if param.requires_grad}
        
        decay = []
        nodecay = []
        
        for name, param in param_dict.items():
            if param.dim() >= 2:
                decay.append(param)
            else:
                nodecay.append(param)
        
        optim_groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": nodecay, "weight_decay": 0.0}
        ]
        
        optimizer = torch.optim.AdamW(
            optim_groups, 
            lr=learning_rate,
            betas=(self.cfg.training.beta1, self.cfg.training.beta2)
        )
        return optimizer

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0

        progress_bar = tqdm(dataloader,desc=f"Training epoch(step: {self.current_step})")
        
        for batch_idx,batch in enumerate(progress_bar):
            x, y = batch["input_ids"].to(self.model.token_embeddings.weight.device), batch["target_ids"].to(self.model.token_embeddings.weight.device)
            
            logits, loss = self.model(x, targets=y)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip)
            
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            self.current_step += 1

            current_lr = self.scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.6f}")
            
            if self.current_step % self.cfg.system.log_every == 0:
                self.save_checkpoint(f"{self.path}/checkpoint_step_{self.current_step}.pt")
        
        return total_loss / len(dataloader)

    def save_checkpoint(self, path):
        import os
        os.makedirs(self.path,exist_ok=True)
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'current_step': self.current_step,
            'config': self.cfg  
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_step = checkpoint['current_step']
        print(f"Checkpoint loaded from {path}")
