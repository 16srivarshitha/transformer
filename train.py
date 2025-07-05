import torch
from config.model_config import ModelConfig
from config.training_config import TrainingConfig
from transformer import EnhancedTransformer
from dataset import create_dataloaders
from trainer import Trainer
from evaluation_metrics import EvaluationMetrics

def main():
    # Configs
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Data
    train_loader, val_loader, tokenizer = create_dataloaders(model_config, training_config)
    print(f'Train batches: {len(train_loader)}, Val batches: {len(val_loader)}')
    
    # Model
    model = EnhancedTransformer(model_config)
    print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
    
    # Trainer
    trainer = Trainer(model, training_config, device)
    
    # Train
    print('Starting training...')
    trainer.train(train_loader, val_loader)
    
    # Evaluation
    evaluator = EvaluationMetrics(tokenizer)
    results = evaluator.evaluate_model(model, val_loader, device)
    
    print('\nFinal Results:')
    print(f'Perplexity: {results["perplexity"]:.4f}')
    print(f'Inference Speed: {results["inference_speed"]:.2f} tokens/sec')
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    print('Final model saved!')

if __name__ == '__main__':
    main()