import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from data import *
from Perceiver import PerceiverModel

import wandb
api_key = "14037597d70b3d9a3bfb20066d401edf14065e6d"
wandb.login(key=api_key)
wandb.init(project="Perceiver autoregressive model", config=config)


def get_optimizer(optimizer_name, model_parameters, lr):
    if optimizer_name == 'SGD':
        return optim.SGD(model_parameters, lr=lr, momentum=0.9)
    elif optimizer_name == 'Adam':
        return optim.Adam(model_parameters, lr=lr)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def train():
    torch.manual_seed(config['seed'])
    device = get_device()

    train_data, val_data, test_data = download_and_extract('http://mattmahoney.net/dc/enwik8.zip', config['data_path'])
    model = PerceiverModel(embed_dim=128, latent_dim=256, heads=8, d_ff=512, seq_len=256, latent_len=256, num_tokens=256, N=6).to(device)
    optimizer = get_optimizer(config['optimizer'], model.parameters(), config['learning_rate'])
    instances_seen = 0
    for i in tqdm.trange(config['num_batches']):
        optimizer.zero_grad()
        input, target = slice_batch(train_data, config['seq_len'], config['batch_size'])
        instances_seen += input.size(0)
        if torch.cuda.is_available():
            input, target = input.cuda(), target.cuda()
            
        tic()    
        output = model(input)

        t = toc()

        loss = F.nll_loss(output.transpose(2, 1), target, reduction='mean')
        
        loss.backward()
        total_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        wandb.log({"Loss/train": loss.item(),
            "Gradient norm": total_norm,
            "Learning Rate": config['learning_rate'],
            "Batch": i,
            "Instances Seen": instances_seen})
        optimizer.step()
        if i != 0 and (i %  config['test_interval'] == 0 or i == config['num_batches'] - 1):
            print(f"Batch {i}, Loss: {loss.item()}")
            
            
            seedfr = random.randint(0, val_data.size(0) - config['seq_len'])
            seed = val_data[seedfr:seedfr + config['seq_len']].to(torch.long)
            
            if torch.cuda.is_available():
                seed = seed.cuda()
                
            generated_text = print_sequence(model, seed, config['char_to_gen'], config['context'])
           
            
            if i == config['num_batches'] - 1:
                torch.save(model.state_dict(), f'saved_models/model_after_batch_{i}.pt')

            upto = test_data.size(0)
            data_sub = test_data[:upto]

            bits_per_byte = estimate_compression(model, data_sub, 10000, context=config['char_to_gen'], batch_size=config['batch_size'] * 2)
            # -- Since we're not computing gradients, we can increase the batch size a little from what we used in
            #    training.

            print(f'epoch{i}: {bits_per_byte:.4} bits per byte')
            wandb.log({"Loss/train": bits_per_byte})

train()