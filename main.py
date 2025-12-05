import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn

from model import Transformer
from trainer import Trainer
from data import get_data_loader

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    print("\n" + "="*50)
    print("Transformer Training with Hydra")
    print("="*50)
    
    print("\nConfiguration:")
    print(OmegaConf.to_yaml(cfg))
    
    torch.manual_seed(cfg.system.seed)
    
    print("\n[1/4] Loading data...")
    
    train_loader, vocab_size, dataset = get_data_loader(
        cfg.data.dataset_name,
        batch_size=cfg.training.batch_size,
        block_size=cfg.data.block_size
    )
    
    if vocab_size > 0:
        cfg.model.vocab_size = vocab_size
        print(f"Vocabulary size from data: {vocab_size}")
    else:
        vocab_size = cfg.model.vocab_size
    
    print("\n[2/4] Creating model...")
    
    model = Transformer(
        num_heads=cfg.model.n_heads,
        d_model=cfg.model.d_model,
        num_layers=cfg.model.n_layers,
        dropout=cfg.model.dropout,
        vocab_size=vocab_size,
        max_seq_length=cfg.model.max_seq_length
    )
    
    device = torch.device(cfg.system.device if torch.cuda.is_available() and cfg.system.device == "cuda" else "cpu")
    model = model.to(device)
    
    print(f"Model on: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n[3/4] Creating trainer...")
    
    if cfg.training.num_training_steps <= 0:
        cfg.training.num_training_steps = cfg.training.num_epochs * len(train_loader)
    
    trainer = Trainer(
        model=model,
        cfg=cfg  
    )
    
    print("\n[4/4] Starting training...")
    print("-" * 50)
    
    for epoch in range(cfg.training.num_epochs):
        print(f"\nEpoch {epoch + 1}/{cfg.training.num_epochs}")
        
    
        avg_loss = trainer.train_epoch(train_loader, epoch)
        
        
        current_lr = trainer.scheduler.get_last_lr()[0] if trainer.scheduler else cfg.training.learning_rate
        
        print(f"  Average Loss: {avg_loss:.4f}, LR: {current_lr:.6f}")
        
        if (epoch + 1) % 2 == 0 or epoch == cfg.training.num_epochs - 1:
            checkpoint_path = f"{cfg.training.checkpoint_dir}/model_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path)
        
        # Generate sample text
        if cfg.system.generate_samples and epoch % 2 == 0:
            print(f"\n  Generating sample...")
            with torch.no_grad():
                prompt = "The meaning of life is"
                tokens = [dataset.stoi.get(c, 0) for c in prompt[:10]]
                if len(tokens) < 2: 
                    tokens = [1, 2, 3, 4, 5]
                
                prompt_tensor = torch.tensor([tokens]).to(device)
                
                generated = model.generate(
                    prompt_tensor,
                    max_new_tokens=30,
                    temperature=0.8,
                    top_k=50
                )
                
                text = dataset.decode(generated[0].cpu().tolist())
                print(f"  Generated: {text[:100]}...")
    
    print("\n" + "="*50)
    print("Training complete!")
    

    final_path = f"{cfg.training.checkpoint_dir}/model_final.pt"
    torch.save(model.state_dict(), final_path)
    print(f"Final model saved to {final_path}")

if __name__ == "__main__":
    main()