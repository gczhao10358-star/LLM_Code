import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from copy import deepcopy
import numpy as np

class PPOConfig:
    def __init__(self):
        self.model_name = "C:/Users/78675/Desktop/大模型面经/GRPO/grpo/model" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.lr = 1e-5
        self.batch_size = 2
        self.max_new_tokens = 32
        self.ppo_epochs = 4
        self.clip_eps = 0.2
        self.kl_coef = 0.1
        self.gamma = 0.99
        self.lam = 0.95

# 1. 价值模型 (Critic): 预测每个 Token 的价值，用于计算 GAE
# 输出: (Batch, Seq_Len)
class Critic(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.value_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        # output_hidden_states=True 确保我们可以拿到 hidden state
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state # (B, L, H)
        values = self.value_head(last_hidden_state).squeeze(-1) # (B, L)
        return values

# 2. 奖励模型 (Reward Model): 给完整的回答打分
# 输出: (Batch, 1) -> 一个标量
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.score_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.last_hidden_state # (B, L, H)
        
        # 计算所有位置的分数
        logits = self.score_head(last_hidden_state).squeeze(-1) # (B, L)
        
        # === 关键：只提取最后一个有效 Token 的分数 ===
        batch_size = input_ids.shape[0]
        scores = []
        for i in range(batch_size):
            # 找到最后一个非 Padding 的位置 (假设 padding=0 in mask)
            # nonzero() 返回索引，[-1] 取最后一个
            last_idx = (attention_mask[i] == 1).nonzero()[-1].item()
            scores.append(logits[i, last_idx])
            
        return torch.stack(scores).unsqueeze(-1) # (B, 1)
    
def init_models(config):
    print(f"正在加载模型: {config.model_name} ...")
    
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    # 生成任务必须左填充
    tokenizer.padding_side = "left" 
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. 加载通用底座 (Base) - 为了省显存，我们先加载一个，然后复制
    raw_model = AutoModelForCausalLM.from_pretrained(config.model_name).to(config.device)
    
    # 2. Actor (策略模型) - 需要训练
    actor = raw_model # 直接用加载的这个作为 Actor
    
    # 3. Reference (参考模型) - 冻结
    ref_model = deepcopy(raw_model)
    ref_model.eval()
    for p in ref_model.parameters(): 
        p.requires_grad = False
    
    # 4. Reward Model (奖励模型) - 使用 Base 结构初始化，通常是冻结的
    # 注意：实际应用中应加载专门训练过的 RM 权重
    rm_base = deepcopy(raw_model.model) # 只取 transformer 部分，不要 lm_head
    reward_model = RewardModel(rm_base).to(config.device)
    reward_model.eval()
    for p in reward_model.parameters(): 
        p.requires_grad = False

    # 5. Critic (价值模型) - 使用 Base 结构初始化，需要训练
    critic_base = deepcopy(raw_model.model)
    critic = Critic(critic_base).to(config.device)

    # 优化器
    opt_actor = Adam(actor.parameters(), lr=config.lr)
    opt_critic = Adam(critic.parameters(), lr=config.lr)

    return actor, ref_model, reward_model, critic, tokenizer, opt_actor, opt_critic


def generate_experience(prompts, actor, tokenizer, config):
    actor.eval()
    
    # 1. Tokenize Prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(config.device)
    prompt_len = inputs.input_ids.shape[1]
    
    # 2. 生成 (Action)
    with torch.no_grad():
        seqs = actor.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            top_k=50,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
    # 3. 构建 Mask
    # Attention Mask: 标记哪些是真实的 Token (非 Padding)
    attention_mask = (seqs != tokenizer.pad_token_id).long()
    
    # Action Mask: 标记哪些是生成的 Response (不包含 Prompt 和 Padding)
    action_mask = torch.zeros_like(attention_mask)
    action_mask[:, prompt_len:] = 1 # Prompt 之后全是 1
    action_mask = action_mask & attention_mask # 去掉 Padding 的部分
    
    return {
        "seqs": seqs,
        "attention_mask": attention_mask,
        "action_mask": action_mask
    }


def compute_advantage(batch, actor, ref_model, reward_model, critic, config):
    seqs = batch['seqs']
    mask = batch['attention_mask']
    act_mask = batch['action_mask']
    
    with torch.no_grad():
        # 1. 计算 Log Probabilities (新旧策略)
        # Actor
        logits = actor(seqs, attention_mask=mask).logits
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        old_log_probs = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # Ref (用于 KL)
        ref_logits = ref_model(seqs, attention_mask=mask).logits
        ref_log_probs = F.log_softmax(ref_logits[:, :-1, :], dim=-1)
        ref_old_log_probs = ref_log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
        
        # 2. 计算 KL 散度 (Token 级别)
        # KL(P||Q) approx logP - logQ
        kl = old_log_probs - ref_old_log_probs 
        
        # 3. 获取模型奖励 (Scalar)
        # reward_score shape: (Batch, 1)
        reward_score = reward_model(seqs, mask)
        
        # 4. 获取 Critic 价值 (Values)
        # values shape: (Batch, Seq_Len)
        values = critic(seqs, mask)
        values = values[:, :-1] # 对齐 Log_probs 的长度
        
        # 5. 组合奖励: R_t = -beta * KL_t (除了最后一个 Token)
        # 最后一个 Token: R_T = -beta * KL_T + Reward_Model_Score
        rewards = -config.kl_coef * kl
        
        batch_size = rewards.shape[0]
        for i in range(batch_size):
            # 找到最后一个有效动作的位置
            last_idx = (act_mask[i] == 1).nonzero()[-1].item() - 1 # log_probs 长度比 seq 少 1
            # 将标量分数加到该位置
            rewards[i, last_idx] += reward_score[i, 0]
            
        # 6. GAE 计算 (倒序递归)
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        seq_len = rewards.shape[1]
        
        for t in reversed(range(seq_len)):
            if t == seq_len - 1:
                next_val = 0
            else:
                next_val = values[:, t + 1]
            
            # TD Error: delta = r + gamma * V_next - V_curr
            delta = rewards[:, t] + config.gamma * next_val - values[:, t]
            
            # GAE = delta + gamma * lambda * GAE_next
            advantages[:, t] = last_gae = delta + config.gamma * config.lam * last_gae
            
        # Returns = Advantage + Value (用于训练 Critic)
        returns = advantages + values
        
    return {
        "seqs": seqs,
        "mask": mask,
        "act_mask": act_mask[:, 1:], # 对齐 log_probs
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "returns": returns,
        "values": values
    }


def train_step(data, actor, critic, opt_actor, opt_critic, config):
    actor.train()
    critic.train()
    
    seqs = data['seqs']
    mask = data['mask']
    act_mask = data['act_mask']
    old_log_probs = data['old_log_probs']
    advantages = data['advantages']
    returns = data['returns']
    
    # 1. 重新计算当前策略的 Log Probs
    logits = actor(seqs, attention_mask=mask).logits
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    new_log_probs = log_probs.gather(dim=-1, index=seqs[:, 1:].unsqueeze(-1)).squeeze(-1)
    
    # 2. Policy Loss (PPO Clip)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1 - config.clip_eps, 1 + config.clip_eps) * advantages
    
    # 只计算生成的 Response 部分的 Loss
    policy_loss = -torch.min(surr1, surr2)
    policy_loss = (policy_loss * act_mask).sum() / act_mask.sum()
    
    # 3. Value Loss
    new_values = critic(seqs, mask)[:, :-1]
    value_loss = (new_values - returns) ** 2
    value_loss = (value_loss * act_mask).sum() / act_mask.sum()
    
    # 4. Backprop
    opt_actor.zero_grad()
    policy_loss.backward()
    opt_actor.step()
    
    opt_critic.zero_grad()
    value_loss.backward()
    opt_critic.step()
    
    return policy_loss.item(), value_loss.item()


def main():
    config = PPOConfig()
    
    # 1. 初始化模型
    actor, ref, reward, critic, tokenizer, opt_a, opt_c = init_models(config)
    
    # 模拟数据
    prompts = [
        "Human: What is 1+1? Assistant:",
        "Human: Write a poem about code. Assistant:",
        "Human: Who are you? Assistant:",
        "Human: Explain PPO algorithm. Assistant:"
    ]
    
    print("=== 开始 PPO 训练 ===")
    
    # 模拟训练循环
    for epoch in range(2): # 大循环
        print(f"Episode {epoch+1}")
        
        # 2. 采样 (Rollout)
        # 这里一次性把所有 prompt 作为一个 batch 简单演示
        exp_data = generate_experience(prompts, actor, tokenizer, config)
        
        # 3. 计算优势 (Evaluate & GAE)
        processed_data = compute_advantage(exp_data, actor, ref, reward, critic, config)
        
        # 4. 参数更新 (PPO Update)
        # PPO 特点：在同一批数据上更新多次 (Epochs)
        for ppo_epoch in range(config.ppo_epochs):
            p_loss, v_loss = train_step(processed_data, actor, critic, opt_a, opt_c, config)
            
            if ppo_epoch % 2 == 0:
                print(f"  Step {ppo_epoch}: Policy Loss={p_loss:.4f}, Value Loss={v_loss:.4f}")
    
    print("训练完成！")

if __name__ == "__main__":
    main()
