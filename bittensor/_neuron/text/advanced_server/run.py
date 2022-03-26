#!/bin/python3
# The MIT License (MIT)
# Copyright © 2021 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
""" Advanced server neuron.

Example:
    $ python miners/text/advanced_server/main.py

"""
import bittensor
import torch
import wandb
import pandas
import datetime
import traceback
import sys
import os

from transformers import get_scheduler

from loguru import logger; logger = logger.opt(colors=True)
from torch.nn.utils import clip_grad_norm_
from datetime import datetime,timedelta
from threading import Lock
from .nucleus_impl import server
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

def serve( 
    config, 
    gp_server: server = None, 
    subtensor = None,
    wallet = None, 
    metagraph = None,
    axon = None
):
    config.to_defaults()

    # Create Subtensor connection
    subtensor = bittensor.subtensor(config = config) if subtensor == None else subtensor

    # Load/Create our bittensor wallet.
    if wallet == None:
        wallet = bittensor.wallet( config = config ).create()
        
    # Instantiate the model we are going to serve on the gp_serverwork.
    # Creating a threading lock for updates to the model
    mutex = Lock()
    
    
    bittensor.tokenizer() 
    timecheck = {}

    # Training Data
    dataset = bittensor.dataset(config=config)

    # load our old model
    if not config.neuron.restart :
        gp_server.load(config.neuron.full_path)
    
    if torch.cuda.device_count() > 1:
        gp_server = torch.nn.DataParallel(gp_server)
    gp_server.to('cuda:0')

    # Create our optimizer.    
    optimizer = torch.optim.AdamW(
        [ {"params": gp_server.parameters()} ],
        lr = 5e-6, # recommended
    )
    num_training_steps = 11390//10 # the number of epochs estimated for myself
    num_warmup_steps = int(0.05 * num_training_steps) # 5% of the number of steps
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )
    
    if config.wandb.api_key != 'default':
        # --- Init Wandb.
        bittensor.wandb(
            config = config,
            cold_pubkey = wallet.coldkeypub.ss58_address,
            hot_pubkey = wallet.hotkey.ss58_address,
            root_dir = config.neuron.full_path
        )

    # -- Main Training loop --
    try:
        interation = 0
        with bittensor.__console__.status('Training...') as status:
            while True:
                # -- download files from the mountain
                data = next(dataset)

                # --- Training step.
                loss, _ = gp_server(data.to('cuda'))
                
                if interation != 0:
                    losses += loss
                else:
                    losses = loss
                
                interation += 1
                status.update(f"training {interation1}/10")
                if interation == 10: # only update the model every 10 iterations
                    interation_ = interation
                    with mutex:
                        logger.info('Backpropagation Started')
                        if interation != 0:
                            losses.backward()
                            interation = 0
                        clip_grad_norm_(gp_server.parameters(), 1.0)
                        
                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        logger.info('Backpropagation Successful: Model updated')
                    
                    wandb_data = {
                        'loss': losses.cpu().item()/interation_ if interation_ != 0 else losses.cpu().item(),
                        "lr": optimizer.param_groups[0]['lr'],
                    }                 

                    bittensor.__console__.print('[green]Current Status:[/green]', wandb_data)

                    # Log losses to wandb.
                    if config.wandb.api_key != 'default':
                        wandb.log( { **wandb_data } )
                    
                    # Save the model
                    gp_server.save(config.neuron.full_path)
            


    except KeyboardInterrupt:
        # --- User ended session ----
        dataset.close()
        
    except Exception as e:
        # --- Unknown error ----
        logger.exception('Unknown exception: {} with traceback {}', e, traceback.format_exc())

