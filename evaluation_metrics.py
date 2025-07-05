import torch
import torch.nn.functional as F
from collections import Counter
import math
import time

class EvaluationMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def calculate_perplexity(self, model, dataloader, device):
        model.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                
                output = model(src, tgt_input)
                loss = F.cross_entropy(
                    output.reshape(-1, output.size(-1)), 
                    tgt_output.reshape(-1), 
                    ignore_index=0
                )
                
                total_loss += loss.item() * (tgt_output != 0).sum().item()
                total_tokens += (tgt_output != 0).sum().item()
        
        return math.exp(total_loss / total_tokens)
    
    def calculate_bleu(self, predictions, references, n=4):
        """Simple BLEU-4 calculation"""
        total_score = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Calculate n-gram precisions
            precisions = []
            for i in range(1, n + 1):
                pred_ngrams = self._get_ngrams(pred_tokens, i)
                ref_ngrams = self._get_ngrams(ref_tokens, i)
                
                if len(pred_ngrams) == 0:
                    precisions.append(0)
                    continue
                    
                matches = sum(min(pred_ngrams[ng], ref_ngrams[ng]) for ng in pred_ngrams)
                precision = matches / len(pred_ngrams)
                precisions.append(precision)
            
            # Brevity penalty
            bp = min(1, math.exp(1 - len(ref_tokens) / len(pred_tokens))) if len(pred_tokens) > 0 else 0
            
            # BLEU score
            if all(p > 0 for p in precisions):
                bleu = bp * math.exp(sum(math.log(p) for p in precisions) / n)
            else:
                bleu = 0
                
            total_score += bleu
        
        return total_score / len(predictions)
    
    def _get_ngrams(self, tokens, n):
        ngrams = Counter()
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams[ngram] += 1
        return ngrams
    
    def measure_inference_speed(self, model, dataloader, device, num_batches=10):
        model.eval()
        total_time = 0
        total_tokens = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if i >= num_batches:
                    break
                    
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                
                start_time = time.time()
                _ = model(src, tgt[:, :-1])
                end_time = time.time()
                
                total_time += (end_time - start_time)
                total_tokens += src.numel()
        
        return total_tokens / total_time  # tokens per second
    
    def evaluate_model(self, model, dataloader, device):
        """Complete evaluation suite"""
        perplexity = self.calculate_perplexity(model, dataloader, device)
        inference_speed = self.measure_inference_speed(model, dataloader, device)
        
        return {
            'perplexity': perplexity,
            'inference_speed': inference_speed,
        }