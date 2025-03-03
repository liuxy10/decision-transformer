import numpy as np
import torch
import os

import time
import tqdm

class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn, scheduler=None, eval_fns=None, ckpt_path=None):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.ckpt_path = (ckpt_path if ckpt_path else None)
            # os.path.join(ckpt_path, 'model.pt') if ckpt_path else None
            
    
        self.highest_reward = -np.inf
        self.highest_success_rate = 0

        self.start_time = time.time()

    def save_checkpoint(self):
        print("[save_checkpoint] model saved at "+self.ckpt_path)
        torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, 'last_model.pt'))
        
    def save_better_checkpoint(self, reward, success_rate):
        if reward > self.highest_reward:
            print("[save_better_checkpoint] currently highest reward model saved at "+self.ckpt_path)
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, 'highest_reward_model.pt'))
            self.highest_reward = reward 

        if success_rate > self.highest_success_rate:
            print("[save_better_checkpoint] currently highest success rate model saved at "+self.ckpt_path)
            torch.save(self.model.state_dict(), os.path.join(self.ckpt_path, 'highest_success_rate_model.pt'))
            self.highest_success_rate = success_rate 


    def train_iteration(self, num_steps, iter_num=0, print_logs=False):

        train_losses = []
        logs = dict()

        train_start = time.time()

        self.model.train()
        for _ in tqdm.tqdm(range(num_steps)):
            train_loss = self.train_step()
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        logs['time/training'] = time.time() - train_start
        if self.ckpt_path:
            self.save_checkpoint()


        eval_start = time.time()

        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        
        
        self.save_better_checkpoint(reward = logs['evaluation/target_400_return_mean'], success_rate=logs['evaluation/target_400_success_rate'])
        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )

        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds, # ok there are reward_preds!
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        

        return loss.detach().cpu().item()
