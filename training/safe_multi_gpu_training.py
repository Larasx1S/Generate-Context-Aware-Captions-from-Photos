import torch
import torch.distributed as dist
import bitsandbytes as bnb
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import (
    Blip2Processor, 
    Blip2ForConditionalGeneration,
    get_cosine_schedule_with_warmup
)
import json
import re
from PIL import Image
import os
import pathlib
from tqdm import tqdm
import logging
import numpy as np
from datetime import datetime
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logmulti.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class GoodNewsDataset(Dataset):
    def __init__(self, json_path, image_base_path, processor, max_samples=None):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        if max_samples:
            self.data = self.data[:max_samples]
            
        self.image_base_path = image_base_path
        self.processor = processor
        
        valid_data_list = []
        
        if dist.is_initialized() and dist.get_rank() == 0:
            logger.info(f"Rank 0: Validating {len(self.data)} images...")
            for item in tqdm(self.data, desc="Validating dataset"):
                img_path = os.path.join(self.image_base_path, item['image'])
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_data_list.append(item)
                    except Exception:
                        logger.warning(f"Skipping corrupted image: {img_path}")
            logger.info(f"Rank 0: Found {len(valid_data_list)} valid images.")

        if dist.is_initialized():
            data_to_broadcast = [valid_data_list] if dist.get_rank() == 0 else [None]
            
            dist.broadcast_object_list(data_to_broadcast, src=0)
            
            self.valid_data = data_to_broadcast[0]
            
            dist.barrier()
            
            if dist.get_rank() != 0:
                    logger.info(f"Rank {dist.get_rank()}: Received {len(self.valid_data)} valid items from rank 0.")
        
        else:
            logger.info("Non-distributed mode: Validating images...")
            for item in tqdm(self.data, desc="Validating dataset"):
                img_path = os.path.join(self.image_base_path, item['image'])
                if os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        valid_data_list.append(item)
                    except Exception:
                        logger.warning(f"Skipping corrupted image: {img_path}")
            self.valid_data = valid_data_list

    def __len__(self):
        return len(self.valid_data)
    
    def __getitem__(self, idx):
        item = self.valid_data[idx]
        img_path = os.path.join(self.image_base_path, item['image'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            caption = item['caption'].strip()
            context = item.get('context', '').strip()
    
            caption = re.sub(r'[^\x00-\x7F]+', '', caption) 
            caption = re.sub(r'\s+', ' ', caption)         
            
            
            input_text = f"Context: {context} Caption: {caption}"
            
            encoding = self.processor(
                images=image,
                text=input_text,
                padding='max_length',
                truncation=True,
                max_length=200,
                return_tensors="pt"
            )
            
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            encoding['labels'] = encoding['input_ids'].clone()
            
            return encoding
            
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def collate_fn(batch):
    if not batch:
        return None
        
    keys = batch[0].keys()
    collated = {}
    
    try:
        for key in keys:
            if key in ['pixel_values', 'input_ids', 'attention_mask', 'labels']:
                collated[key] = torch.stack([item[key] for item in batch])
        return collated
    except Exception as e:
        logger.error(f"Collate error: {e}")
        return None

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        return False, 0, 1, 0
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend='nccl')
    
    return True, rank, world_size, local_rank

def setup_trainable_params_efficient(model, rank, output_dir):
    if rank == 0:
        logger.info("Setting up trainable parameters...")

    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        all_param_names_file = output_dir / "all_param_names.txt"
        
        logger.info("\n" + "="*70)
        logger.info("DIAGNOSTIC 2.0: Writing ALL parameter names to find the correct name...")
        
        with open(all_param_names_file, "w") as f:
            for name, _ in model.named_parameters():
                f.write(name + "\n")
        
        logger.info(f"===> All parameter names have been written to the file: {all_param_names_file}")
        logger.info("===> PLEASE OPEN THIS FILE and search for the name of the last layer.")
        logger.info("="*70 + "\n")

    trainable_params = []
    total_params = 0
    
    num_decoder_layers = model.language_model.config.num_hidden_layers
    last_n_layers = 1
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        should_train = False
        reason = ""
        
        if "qformer" in name.lower():
            should_train = True
            reason = "Q_FORMER"
        
        elif "lm_head" in name: 
            should_train = True
            reason = "LM_HEAD"
            
        elif "language_model.model.decoder.layers." in name:
            try:
                layer_str = name.split('language_model.model.decoder.layers.')[1].split('.')[0]
                if layer_str.isdigit():
                    layer_num = int(layer_str)
                    if layer_num >= (num_decoder_layers - last_n_layers):
                        should_train = True
                        reason = f"OPT_DECODER_LAYER_{layer_num}"
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Could not parse layer number from {name}: {e}")

        param.requires_grad = should_train
        if should_train:
            trainable_params.append(name)
            if rank == 0:
                logger.info(f"✓ TRAINABLE: {name} ({reason})")

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if rank == 0:
        logger.info("\n" + "="*60)
        logger.info("PARAMETER SUMMARY")
        logger.info("="*60)
        logger.info(f"Trainable parameters: {trainable_count:,} / {total_params:,}")
        logger.info(f"Percentage trainable: {100*trainable_count/total_params:.2f}%")
        logger.info(f"Strategy: Q-Former + LM Head + Last {last_n_layers} OPT Decoder Layers")
        logger.info("="*60)
        
    return trainable_params

def evaluate(model, dataloader, device, rank):
    model.eval()
    total_loss = 0
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=rank!=0):
            if batch is None:
                continue
                
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    total_loss += loss.item()
                    valid_batches += 1
            except Exception as e:
                if rank == 0:
                    logger.warning(f"Evaluation error: {e}")
                continue
    
    if dist.is_initialized():
        total_loss_tensor = torch.tensor(total_loss, device=device)
        valid_batches_tensor = torch.tensor(valid_batches, device=device)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(valid_batches_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = total_loss_tensor.item()
        valid_batches = valid_batches_tensor.item()
    
    return total_loss / max(valid_batches, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    
    is_distributed, rank, world_size, local_rank = setup_distributed()
    
    script_dir = pathlib.Path(__file__).parent
    project_root = script_dir.parent
    
    train_json = project_root / "data" / "processed" / "train_224.json"
    val_json = project_root / "data" / "processed" / "val_224.json"
    test_json = project_root / "data" / "processed" / "test_224.json"
    image_base_path = project_root / "data" / "images"
    output_dir = project_root / "results" / "multi_gpu_training"
    
    model_name = "Salesforce/blip2-opt-2.7b" 
    
    batch_size = 1 if is_distributed else 1
    gradient_accumulation_steps = 64
    learning_rate = 3e-5
    min_learning_rate = 8e-6
    weight_decay = 0.05
    num_epochs = 12
    warmup_ratio = 0.2
    max_grad_norm = 0.5
    
    save_steps = 400
    eval_steps = 400
    logging_steps = 20
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"multi_gpu_{timestamp}"
    
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("="*70)
        logger.info("BLIP2 MULTI-GPU TRAINING")
        logger.info("="*70)
        logger.info(f"Model: {model_name}")
        logger.info(f"GPUs: {world_size} (Rank {rank})")
        logger.info(f"Batch size per GPU: {batch_size}")
        logger.info(f"Effective batch size: {batch_size * world_size * gradient_accumulation_steps}")
        logger.info(f"Learning rate: {learning_rate}")
        logger.info(f"Output dir: {output_dir}")
        logger.info("="*70)
    
    torch.cuda.empty_cache()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        logger.info(f"Loading model: {model_name}")
    
    processor = Blip2Processor.from_pretrained(model_name)
    
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=None
    )
    
    model = model.to(device)
    
    model.gradient_checkpointing_enable()
    
    trainable_params = setup_trainable_params_efficient(model, rank, output_dir)
    
    if len(trainable_params) == 0:
        if rank == 0:
            logger.error("No trainable parameters found! Exiting.")
        return
    
    if is_distributed:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    if rank == 0:
        logger.info("Loading datasets...")
    
    train_dataset = GoodNewsDataset(train_json, image_base_path, processor)
    val_dataset = GoodNewsDataset(val_json, image_base_path, processor, max_samples=1000)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=False
    )
    
    if is_distributed:
        trainable_model_params = filter(lambda p: p.requires_grad, model.module.parameters())
    else:
        trainable_model_params = filter(lambda p: p.requires_grad, model.parameters())
        
    optimizer = bnb.optim.AdamW8bit(
        trainable_model_params,
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=weight_decay
    )
    
    steps_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    if rank == 0:
        logger.info(f"Steps per epoch: {steps_per_epoch}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Warmup steps: {warmup_steps}")
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
        num_cycles=0.5
    )
    
    if rank == 0:
        logger.info("Starting training...")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
            
        if rank == 0:
            logger.info(f"\nEPOCH {epoch + 1}/{num_epochs}")
        
        model.train()
        epoch_loss = 0
        valid_steps = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}", disable=rank!=0)
        optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            if batch is None:
                continue
                
            try:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    outputs = model(**batch)
                    loss = outputs.loss / gradient_accumulation_steps
                
                if torch.isnan(loss) or torch.isinf(loss):
                    if rank == 0:
                        logger.warning(f"NaN/Inf detected at step {global_step}! Skipping batch.")
                    optimizer.zero_grad()
                    continue
                
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    if is_distributed:
                        torch.nn.utils.clip_grad_norm_(model.module.parameters(), max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    val_loss = None 
                    if global_step % eval_steps == 0:
                        val_loss = evaluate(model, val_loader, device, rank)
                        model.train()
                    
                    if rank == 0:
                        current_lr = scheduler.get_last_lr()[0]

                        if global_step % logging_steps == 0:
                            logger.info(
                                f"Step {global_step}, "
                                f"Loss: {loss.item() * gradient_accumulation_steps:.4f}, "
                                f"LR: {current_lr:.2e}"
                            )
                        
                        if global_step % save_steps == 0:
                            checkpoint_dir = output_dir / f"checkpoint-{global_step}"
                            save_model = model.module if is_distributed else model
                            save_model.save_pretrained(checkpoint_dir)
                            processor.save_pretrained(checkpoint_dir)
                            logger.info(f"Saved checkpoint: {checkpoint_dir}")
                        
                        if val_loss is not None:
                            logger.info(f"Validation loss: {val_loss:.4f}")
                            
                            if val_loss < best_val_loss:
                                best_val_loss = val_loss
                                best_dir = output_dir / "best_model"
                                save_model = model.module if is_distributed else model
                                save_model.save_pretrained(best_dir)
                                processor.save_pretrained(best_dir)
                                logger.info(f"New best model! Loss: {val_loss:.4f}")
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                valid_steps += 1
                
                if rank == 0 and valid_steps > 0:
                    avg_loss = epoch_loss / valid_steps
                    current_lr = scheduler.get_last_lr()[0]
                    progress_bar.set_postfix({
                        'loss': f'{loss.item() * gradient_accumulation_steps:.4f}',
                        'avg': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}'
                    })
                
            except Exception as e:
                if rank == 0:
                    logger.error(f"Training error at step {step}: {e}")
                optimizer.zero_grad()
                continue
        
        if rank == 0:
            avg_epoch_loss = epoch_loss / max(valid_steps, 1)
            logger.info(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss:.4f}")
            
            epoch_dir = output_dir / f"epoch-{epoch + 1}"
            save_model = model.module if is_distributed else model
            save_model.save_pretrained(epoch_dir)
            processor.save_pretrained(epoch_dir)
            logger.info(f"Saved epoch checkpoint: {epoch_dir}")
    
    if rank == 0:
        logger.info("\n" + "="*70)
        logger.info("TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"Best validation loss: {best_val_loss:.4f}")
        logger.info(f"Best model saved at: {output_dir / 'best_model'}")
        logger.info("="*70)
    
    if is_distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()