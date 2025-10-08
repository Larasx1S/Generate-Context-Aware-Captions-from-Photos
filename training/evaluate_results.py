import json
import re
from collections import Counter
from rouge_score import rouge_scorer
import os

class SimpleCaptionEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def simple_tokenize(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [token for token in tokens if token]
    
    def compute_bleu_ngrams(self, reference_tokens, hypothesis_tokens, n):
        if len(hypothesis_tokens) < n:
            return 0.0
        
        ref_ngrams = Counter()
        hyp_ngrams = Counter()
        
        for i in range(len(reference_tokens) - n + 1):
            ngram = tuple(reference_tokens[i:i+n])
            ref_ngrams[ngram] += 1
            
        for i in range(len(hypothesis_tokens) - n + 1):
            ngram = tuple(hypothesis_tokens[i:i+n])
            hyp_ngrams[ngram] += 1
        
        matches = 0
        total = 0
        
        for ngram in hyp_ngrams:
            matches += min(hyp_ngrams[ngram], ref_ngrams.get(ngram, 0))
            total += hyp_ngrams[ngram]
        
        return matches / total if total > 0 else 0.0
    
    def compute_bleu(self, references, hypotheses):
        bleu_scores = {}
        
        for n in range(1, 5):
            precisions = []
            
            for ref, hyp in zip(references, hypotheses):
                ref_tokens = self.simple_tokenize(ref)
                hyp_tokens = self.simple_tokenize(hyp)
                
                precision = self.compute_bleu_ngrams(ref_tokens, hyp_tokens, n)
                precisions.append(precision)
            
            avg_precision = sum(precisions) / len(precisions) if precisions else 0.0
            bleu_scores[f'BLEU-{n}'] = avg_precision * 100
        
        return bleu_scores
    
    def compute_rouge(self, references, hypotheses):
        rouge_scores = {'ROUGE-1': [], 'ROUGE-2': [], 'ROUGE-L': []}
        
        for ref, hyp in zip(references, hypotheses):
            try:
                scores = self.rouge_scorer.score(ref, hyp)
                rouge_scores['ROUGE-1'].append(scores['rouge1'].fmeasure)
                rouge_scores['ROUGE-2'].append(scores['rouge2'].fmeasure)
                rouge_scores['ROUGE-L'].append(scores['rougeL'].fmeasure)
            except:
                rouge_scores['ROUGE-1'].append(0.0)
                rouge_scores['ROUGE-2'].append(0.0)
                rouge_scores['ROUGE-L'].append(0.0)
        
        return {
            'ROUGE-1': sum(rouge_scores['ROUGE-1']) / len(rouge_scores['ROUGE-1']) * 100,
            'ROUGE-2': sum(rouge_scores['ROUGE-2']) / len(rouge_scores['ROUGE-2']) * 100,
            'ROUGE-L': sum(rouge_scores['ROUGE-L']) / len(rouge_scores['ROUGE-L']) * 100
        }
    
    def compute_length_stats(self, references, hypotheses):
        ref_lengths = [len(self.simple_tokenize(ref)) for ref in references]
        hyp_lengths = [len(self.simple_tokenize(hyp)) for hyp in hypotheses]
        
        return {
            'avg_ref_length': sum(ref_lengths) / len(ref_lengths),
            'avg_hyp_length': sum(hyp_lengths) / len(hyp_lengths),
            'length_ratio': (sum(hyp_lengths) / sum(ref_lengths)) if sum(ref_lengths) > 0 else 0.0
        }

def evaluate_captions_simple(results_file, output_file=None):
    print(f"Loading results from: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    references = []
    hypotheses = []
    
    for result in results:
        references.append(result.get('original_caption', ''))
        
        generated_full = result.get('generated_caption', '')
        if "Caption:" in generated_full:
            hypothesis = generated_full.split("Caption:", 1)[1].strip()
        else:
            hypothesis = generated_full.strip()
        hypotheses.append(hypothesis)

    print(f"Evaluating {len(references)} caption pairs...")
    
    evaluator = SimpleCaptionEvaluator()
    all_scores = {}
    
    print("Computing BLEU scores...")
    bleu_scores = evaluator.compute_bleu(references, hypotheses)
    all_scores.update(bleu_scores)
    
    print("Computing ROUGE scores...")
    try:
        rouge_scores = evaluator.compute_rouge(references, hypotheses)
        all_scores.update(rouge_scores)
    except Exception as e:
        print(f"ROUGE computation failed: {e}")
        all_scores.update({'ROUGE-1': 0.0, 'ROUGE-2': 0.0, 'ROUGE-L': 0.0})
    
    length_stats = evaluator.compute_length_stats(references, hypotheses)
    all_scores.update(length_stats)
    
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"BLEU-1: {all_scores['BLEU-1']:.2f}%")
    print(f"BLEU-2: {all_scores['BLEU-2']:.2f}%")
    print(f"BLEU-3: {all_scores['BLEU-3']:.2f}%")
    print(f"BLEU-4: {all_scores['BLEU-4']:.2f}%")
    print(f"ROUGE-1: {all_scores['ROUGE-1']:.2f}%")
    print(f"ROUGE-2: {all_scores['ROUGE-2']:.2f}%")
    print(f"ROUGE-L: {all_scores['ROUGE-L']:.2f}%")
    
    print("\n" + "-"*30)
    print("LENGTH STATISTICS:")
    print(f"Avg Reference Length: {all_scores['avg_ref_length']:.1f} words")
    print(f"Avg Generated Length: {all_scores['avg_hyp_length']:.1f} words")
    print(f"Length Ratio: {all_scores['length_ratio']:.2f}")
    
    print("\n" + "="*50)
    print("KEY METRICS FOR REPORTING:")
    print("="*50)
    print(f"BLEU-4: {all_scores['BLEU-4']:.2f}%")
    print(f"ROUGE-L: {all_scores['ROUGE-L']:.2f}%")

    print("\n" + "="*50)
    print("CAPTION EXAMPLES:")
    print("="*50)
    
    for i, result in enumerate(results[:3]):
        generated_full = result.get('generated_caption', '')
        if "Caption:" in generated_full:
            hypothesis = generated_full.split("Caption:", 1)[1].strip()
        else:
            hypothesis = generated_full.strip()
            
        print(f"\nExample {i+1}:")
        print(f"Original:  {result.get('original_caption', '')}")
        print(f"Generated: {hypothesis}")
        if result.get('context'):
            print(f"Context:   {result.get('context')}")
            
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_scores, f, indent=2)
        print(f"\nScores saved to: {output_file}")
    
    return all_scores

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Simplified evaluation for news image captioning')
    parser.add_argument('--results_file', type=str, required=True,
                        help='Path to results JSON file')
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to save evaluation scores')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_file):
        print(f"Error: Results file not found: {args.results_file}")
        return
    
    evaluate_captions_simple(args.results_file, args.output_file)

if __name__ == "__main__":
    main()