import json
import os
from collections import defaultdict
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, cohen_kappa_score
from pathlib import Path

class ResultsProcessor:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = defaultdict(lambda: defaultdict(list))
        
    def should_filter_question(self, question_data):
        """Filter out questions based on criteria"""
        # Check if question contains "all" or "unknown"
        question_text = question_data.get('question', '').lower()
        if 'all' in question_text or 'unknown' in question_text:
            return True
        
        # Check if answer options contain "all" or "unknown"
        options = question_data.get('options', [])
        answer_idx = question_data.get('answer_index', -1)
        if answer_idx >= 0 and answer_idx < len(options):
            answer_text = options[answer_idx].lower()
            if 'all' in answer_text or 'unknown' in answer_text:
                return True
        
        # Check all options for "all" or "unknown"
        for option in options:
            if option.lower() in ['all', 'unknown']:
                return True
                
        return False
    
    def extract_answer_letter(self, response):
        """
        Enhanced answer extraction that handles multiple response formats
        
        Handles cases like:
        - "A"
        - ["B"] (list format from some models like Qwen)
        - "The answer is B"
        - "B. Central Perk"
        - "I think the answer is C because..."
        - "Answer: D"
        - Various other formats
        """
        import re
        
        if not response:
            return None
        
        # Handle list responses (e.g., ["B"] from Qwen)
        if isinstance(response, list):
            if len(response) > 0:
                response = response[0]
            else:
                return None
        
        response = str(response).strip()
        if not response:
            return None
        
        # Method 1: Check if first character is A/B/C/D
        first_char = response[0].upper()
        if first_char in ['A', 'B', 'C', 'D']:
            return first_char
        
        # Method 2: Look for patterns like "answer is X" or "Answer: X"
        patterns = [
            r'answer\s+is\s+([A-D])',
            r'answer:\s*([A-D])',
            r'select\s+([A-D])',
            r'option\s+([A-D])',
            r'choose\s+([A-D])',
            r'\b([A-D])\s*[\.\):]',  # Matches "A.", "B)", "C:"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # Method 3: Find any standalone A, B, C, or D
        matches = re.findall(r'\b([A-D])\b', response.upper())
        if matches:
            # Return the first match
            return matches[0]
        
        # Method 4: Check if response contains exactly one of A/B/C/D anywhere
        letters_found = [letter for letter in ['A', 'B', 'C', 'D'] if letter in response.upper()]
        if len(letters_found) == 1:
            return letters_found[0]
        
        return None
    
    def check_duplicate_options(self, question_data):
        """Check if there are duplicate options with same content"""
        options = question_data.get('options', [])
        option_map = defaultdict(list)
        
        for idx, option in enumerate(options):
            option_clean = option.strip().lower()
            option_map[option_clean].append(chr(65 + idx))  # A, B, C, D
        
        return option_map
    
    def is_correct(self, question_data):
        """Check if the model's answer is correct, handling special cases"""
        result = question_data.get('result', {})
        model_response = result.get('model_response', '')
        ground_truth = result.get('ground_truth_answer', '')
        
        # Extract answer letter from model response
        model_answer = self.extract_answer_letter(model_response)
        if model_answer is None:
            return False
        
        # Simple case: direct match
        if model_answer == ground_truth:
            return True
        
        # Check for duplicate options
        option_map = self.check_duplicate_options(question_data)
        options = question_data.get('options', [])
        
        # Get the ground truth option text
        gt_idx = ord(ground_truth) - 65  # A=0, B=1, etc.
        if gt_idx < 0 or gt_idx >= len(options):
            return False
        
        gt_option_text = options[gt_idx].strip().lower()
        
        # Get the model answer option text
        model_idx = ord(model_answer) - 65
        if model_idx < 0 or model_idx >= len(options):
            return False
        
        model_option_text = options[model_idx].strip().lower()
        
        # Check if they have the same content
        if gt_option_text == model_option_text:
            return True
        
        return False
    
    def compute_metrics(self, questions):
        """Compute accuracy, precision, recall, F1, and Cohen's kappa"""
        if not questions:
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'precision_weighted': 0.0,
                'recall_macro': 0.0,
                'recall_weighted': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'kappa': 0.0
            }
        
        y_true = []
        y_pred = []
        
        for q in questions:
            result = q.get('result', {})
            ground_truth = result.get('ground_truth_answer', '')
            model_response = result.get('model_response', '')
            
            # Convert to indices
            gt_idx = ord(ground_truth) - 65 if ground_truth and len(ground_truth) == 1 else -1
            
            model_answer = self.extract_answer_letter(model_response)
            pred_idx = ord(model_answer) - 65 if model_answer else -1
            
            if gt_idx >= 0:
                y_true.append(gt_idx)
                y_pred.append(pred_idx if pred_idx >= 0 else -1)
        
        if not y_true:
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'precision_macro': 0.0,
                'precision_weighted': 0.0,
                'recall_macro': 0.0,
                'recall_weighted': 0.0,
                'f1_macro': 0.0,
                'f1_weighted': 0.0,
                'kappa': 0.0
            }
        
        # Calculate metrics
        correct = sum([1 for q in questions if self.is_correct(q)])
        accuracy = correct / len(questions)
        
        # For precision, recall, F1
        try:
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                y_true, y_pred, average='weighted', zero_division=0
            )
        except:
            precision_macro = recall_macro = f1_macro = 0.0
            precision_weighted = recall_weighted = f1_weighted = 0.0
        
        # Cohen's Kappa
        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except:
            kappa = 0.0
        
        return {
            'total': len(questions),
            'correct': correct,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'precision_weighted': precision_weighted,
            'recall_macro': recall_macro,
            'recall_weighted': recall_weighted,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'kappa': kappa
        }
    
    def process_json_file(self, filepath):
        """Process a single JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Handle both formats:
        # 1. {"qa_dataset": [...]} - dict format
        # 2. [...] - direct list format
        if isinstance(data, dict):
            qa_dataset = data.get('qa_dataset', [])
        elif isinstance(data, list):
            qa_dataset = data
        else:
            print(f"Warning: Unexpected JSON format in {filepath}")
            return []
        
        # Filter questions
        filtered_questions = [q for q in qa_dataset if not self.should_filter_question(q)]
        
        return filtered_questions
    
    def categorize_questions(self, questions):
        """Categorize questions by various attributes"""
        categories = {
            'episode': defaultdict(list),
            'question_type': defaultdict(list),
        }
        
        for q in questions:
            # Episode
            episode_span = q.get('episode_span', [])
            if episode_span:
                episode_id = episode_span[0]
                categories['episode'][episode_id].append(q)
            
            # Question type - map to main categories
            q_type = q.get('question_type', 'unknown')
            
            # Map question types to main categories
            if q_type == 'single target recall':
                main_type = 'Single Target Recall'
            elif q_type == 'boolean':
                main_type = 'Boolean'
            elif q_type == 'temporal: chronological ordering':
                main_type = 'Temporal: Chronological Ordering'
            elif q_type == 'temporal: latest event retrieval':
                main_type = 'Temporal: Latest Event Retrieval'
            elif q_type == 'temporal: single cue retrieval':
                main_type = 'Temporal: Location Sequence Recognition'
            else:
                main_type = q_type
            
            categories['question_type'][main_type].append(q)
        
        return categories
    
    def format_summary(self, overall_metrics, category_metrics, title="EVALUATION SUMMARY"):
        """Format summary in the required text format"""
        lines = []
        lines.append("=" * 100)
        lines.append(title)
        lines.append("=" * 100)
        
        # Overall performance
        lines.append(f"Total Questions: {overall_metrics['total']}")
        lines.append(f"Correct Answers: {overall_metrics['correct']}")
        lines.append(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        lines.append("")
        lines.append("Detailed Metrics:")
        lines.append(f"  Precision (Macro):    {overall_metrics['precision_macro']:.4f}")
        lines.append(f"  Precision (Weighted): {overall_metrics['precision_weighted']:.4f}")
        lines.append(f"  Recall (Macro):       {overall_metrics['recall_macro']:.4f}")
        lines.append(f"  Recall (Weighted):    {overall_metrics['recall_weighted']:.4f}")
        lines.append(f"  F1 Score (Macro):     {overall_metrics['f1_macro']:.4f}")
        lines.append(f"  F1 Score (Weighted):  {overall_metrics['f1_weighted']:.4f}")
        lines.append(f"  Cohen's Kappa:        {overall_metrics['kappa']:.4f}")
        lines.append("")
        
        # Category breakdowns
        for category_name in ['question_type', 'episode']:
            if category_name not in category_metrics:
                continue
            
            lines.append("=" * 100)
            lines.append(f"PERFORMANCE BY {category_name.upper().replace('_', ' ')}")
            lines.append("=" * 100)
            
            # Header
            lines.append(f"{'Category':<40} {'Total':<8} {'Correct':<8} {'Acc':<8} {'P(M)':<8} {'R(M)':<8} {'F1(M)':<8} {'Kappa':<8}")
            lines.append("-" * 100)
            
            # Sort categories
            sorted_cats = sorted(category_metrics[category_name].items())
            
            for cat_name, metrics in sorted_cats:
                lines.append(
                    f"{cat_name:<40} {metrics['total']:<8} {metrics['correct']:<8} "
                    f"{metrics['accuracy']:<8.4f} {metrics['precision_macro']:<8.4f} "
                    f"{metrics['recall_macro']:<8.4f} {metrics['f1_macro']:<8.4f} "
                    f"{metrics['kappa']:<8.4f}"
                )
            
            lines.append("")
        
        return "\n".join(lines)
    
    def process_all_files(self):
        """Process all JSON files in the directory structure"""
        results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        
        print(f"\nSearching in base path: {self.base_path}")
        print("-" * 80)
        
        # Navigate through results_vanilla and results_goldfish
        for result_type in ['results_vanilla', 'results_goldfish']:
            result_path = os.path.join(self.base_path, result_type)
            
            print(f"\nChecking: {result_path}")
            
            if not os.path.exists(result_path):
                print(f"  ❌ Path does not exist: {result_path}")
                continue
            
            print(f"  ✓ Path exists")
            
            # Get all model directories
            try:
                model_dirs = [d for d in os.listdir(result_path) if os.path.isdir(os.path.join(result_path, d))]
                print(f"  Found {len(model_dirs)} model directories: {model_dirs}")
            except Exception as e:
                print(f"  ❌ Error listing directory: {e}")
                continue
            
            for model_name in model_dirs:
                model_path = os.path.join(result_path, model_name)
                
                print(f"\n  Model: {model_name}")
                
                # For vanilla: with_context and without_context come BEFORE runs
                if result_type == 'results_vanilla':
                    # Check for with_context and without_context at model level
                    for context_dir in ['with_context', 'without_context']:
                        context_path = os.path.join(model_path, context_dir)
                        
                        if not os.path.exists(context_path) or not os.path.isdir(context_path):
                            print(f"    {context_dir}: directory not found")
                            continue
                        
                        print(f"    Context: {context_dir}")
                        
                        # Get run directories inside context directory
                        try:
                            run_dirs = [d for d in os.listdir(context_path) 
                                       if os.path.isdir(os.path.join(context_path, d)) and d.startswith('run_')]
                            print(f"      Found {len(run_dirs)} run directories: {run_dirs}")
                        except Exception as e:
                            print(f"      ❌ Error listing runs: {e}")
                            continue
                        
                        for run_name in run_dirs:
                            run_path = os.path.join(context_path, run_name)
                            
                            print(f"      Run: {run_name}")
                            
                            # Get all JSON files in this run directory
                            json_files = [f for f in os.listdir(run_path) if f.endswith('.json')]
                            print(f"        JSON files: {len(json_files)}")
                            
                            if json_files:
                                all_questions = []
                                for json_file in json_files:
                                    json_path = os.path.join(run_path, json_file)
                                    questions = self.process_json_file(json_path)
                                    all_questions.extend(questions)
                                
                                if all_questions:
                                    key = (model_name, run_name, context_dir, 'vanilla')
                                    results[key] = all_questions
                                    print(f"          ✓ Added {len(all_questions)} questions for {key}")
                
                # For goldfish: run directories come first, then caption types
                elif result_type == 'results_goldfish':
                    # Get all run directories
                    try:
                        run_dirs = [d for d in os.listdir(model_path) 
                                   if os.path.isdir(os.path.join(model_path, d)) and d.startswith('run_')]
                        print(f"    Found {len(run_dirs)} run directories: {run_dirs}")
                    except Exception as e:
                        print(f"    ❌ Error listing runs: {e}")
                        continue
                    
                    for run_name in run_dirs:
                        run_path = os.path.join(model_path, run_name)
                        
                        print(f"    Run: {run_name}")
                        for caption_type in os.listdir(run_path):
                            caption_path = os.path.join(run_path, caption_type)
                            
                            if not os.path.isdir(caption_path):
                                continue
                            
                            # Determine context type
                            if 'generic' in caption_type:
                                context_type = 'without_context'
                            elif 'specific' in caption_type:
                                context_type = 'with_context'
                            else:
                                continue
                            
                            print(f"      {caption_type} -> {context_type}")
                            
                            # Get nn directories
                            for nn_dir in os.listdir(caption_path):
                                nn_path = os.path.join(caption_path, nn_dir)
                                
                                if not os.path.isdir(nn_path):
                                    continue
                                
                                # Extract nn value
                                nn_value = nn_dir  # nn_1, nn_3, nn_5
                                
                                # Get all JSON files
                                json_files = [f for f in os.listdir(nn_path) if f.endswith('.json')]
                                print(f"        {nn_dir}: {len(json_files)} JSON files")
                                
                                all_questions = []
                                for json_file in json_files:
                                    json_path = os.path.join(nn_path, json_file)
                                    questions = self.process_json_file(json_path)
                                    all_questions.extend(questions)
                                
                                if all_questions:
                                    key = (model_name, run_name, context_type, nn_value)
                                    results[key] = all_questions
                                    print(f"          ✓ Added {len(all_questions)} questions for {key}")
        
        print("\n" + "=" * 80)
        print(f"Total configurations found: {len(results)}")
        print("=" * 80)
        
        return results
    
    def generate_summaries(self, results, output_dir):
        """Generate all required summary files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Store all metrics for averaging
        all_metrics = defaultdict(lambda: defaultdict(list))
        
        # Generate individual summaries for each configuration
        for key, questions in results.items():
            model_name, run_name, context_type, nn_value = key
            
            # Compute overall metrics
            overall_metrics = self.compute_metrics(questions)
            
            # Categorize and compute metrics
            categories = self.categorize_questions(questions)
            category_metrics = {}
            
            for cat_type, cat_dict in categories.items():
                category_metrics[cat_type] = {}
                for cat_name, cat_questions in cat_dict.items():
                    category_metrics[cat_type][cat_name] = self.compute_metrics(cat_questions)
            
            # Format and save summary
            if nn_value == 'vanilla':
                title = f"{model_name} - {run_name} - {context_type} - Vanilla"
                filename = f"{model_name}_{run_name}_{context_type}_vanilla.txt"
            else:
                title = f"{model_name} - {run_name} - {context_type} - {nn_value}"
                filename = f"{model_name}_{run_name}_{context_type}_{nn_value}.txt"
            
            summary_text = self.format_summary(overall_metrics, category_metrics, title)
            
            # Save to file
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(summary_text)
            
            # Store for averaging
            config_key = (model_name, context_type, nn_value)
            all_metrics[config_key]['overall'].append(overall_metrics)
            
            for cat_type, cat_dict in category_metrics.items():
                for cat_name, metrics in cat_dict.items():
                    all_metrics[config_key][f"{cat_type}_{cat_name}"].append(metrics)
        
        return all_metrics
    
    def average_metrics(self, metrics_list):
        """Average metrics across multiple runs and calculate standard deviation"""
        if not metrics_list:
            return {
                'total': 0,
                'correct': 0,
                'accuracy': 0.0,
                'accuracy_std': 0.0,
                'precision_macro': 0.0,
                'precision_macro_std': 0.0,
                'precision_weighted': 0.0,
                'precision_weighted_std': 0.0,
                'recall_macro': 0.0,
                'recall_macro_std': 0.0,
                'recall_weighted': 0.0,
                'recall_weighted_std': 0.0,
                'f1_macro': 0.0,
                'f1_macro_std': 0.0,
                'f1_weighted': 0.0,
                'f1_weighted_std': 0.0,
                'kappa': 0.0,
                'kappa_std': 0.0
            }
        
        avg_metrics = {
            'total': int(np.mean([m['total'] for m in metrics_list])),
            'correct': int(np.mean([m['correct'] for m in metrics_list])),
            'accuracy': np.mean([m['accuracy'] for m in metrics_list]),
            'accuracy_std': np.std([m['accuracy'] for m in metrics_list]),
            'precision_macro': np.mean([m['precision_macro'] for m in metrics_list]),
            'precision_macro_std': np.std([m['precision_macro'] for m in metrics_list]),
            'precision_weighted': np.mean([m['precision_weighted'] for m in metrics_list]),
            'precision_weighted_std': np.std([m['precision_weighted'] for m in metrics_list]),
            'recall_macro': np.mean([m['recall_macro'] for m in metrics_list]),
            'recall_macro_std': np.std([m['recall_macro'] for m in metrics_list]),
            'recall_weighted': np.mean([m['recall_weighted'] for m in metrics_list]),
            'recall_weighted_std': np.std([m['recall_weighted'] for m in metrics_list]),
            'f1_macro': np.mean([m['f1_macro'] for m in metrics_list]),
            'f1_macro_std': np.std([m['f1_macro'] for m in metrics_list]),
            'f1_weighted': np.mean([m['f1_weighted'] for m in metrics_list]),
            'f1_weighted_std': np.std([m['f1_weighted'] for m in metrics_list]),
            'kappa': np.mean([m['kappa'] for m in metrics_list]),
            'kappa_std': np.std([m['kappa'] for m in metrics_list])
        }
        
        return avg_metrics
    
    def generate_final_summary(self, all_metrics, output_dir):
        """Generate final summary with averaged metrics across runs"""
        # Average metrics for each configuration
        averaged_metrics = {}
        
        for config_key, metrics_dict in all_metrics.items():
            model_name, context_type, nn_value = config_key
            averaged_metrics[config_key] = {}
            
            for metric_type, metrics_list in metrics_dict.items():
                averaged_metrics[config_key][metric_type] = self.average_metrics(metrics_list)
        
        # Create comparison table
        lines = []
        lines.append("=" * 150)
        lines.append("FINAL SUMMARY - AVERAGED ACROSS 3 RUNS")
        lines.append("=" * 150)
        lines.append("")
        
        # Overall comparison table
        lines.append("OVERALL PERFORMANCE COMPARISON")
        lines.append("-" * 180)
        
        # Table header - adjust for vanilla not having nn, and add std columns
        header = f"{'Model':<25} {'Context':<20} {'Config':<15} {'Acc':>12} {'F1':>12} {'Kappa':>12}"
        lines.append(header)
        lines.append("-" * 180)
        
        # Sort configurations
        sorted_configs = sorted(averaged_metrics.keys())
        
        for config_key in sorted_configs:
            model_name, context_type, nn_value = config_key
            metrics = averaged_metrics[config_key].get('overall', {})
            
            # Format config column - show vanilla as-is, goldfish with nn value
            if nn_value == 'vanilla':
                config_display = 'Vanilla'
            else:
                config_display = nn_value  # nn_1, nn_3, nn_5
            
            # Format with mean ± std
            acc_str = f"{metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f}"
            f1_str = f"{metrics['f1_macro']:.4f}±{metrics['f1_macro_std']:.4f}"
            kappa_str = f"{metrics['kappa']:.4f}±{metrics['kappa_std']:.4f}"
            
            row = f"{model_name:<25} {context_type:<20} {config_display:<15} {acc_str:>12} {f1_str:>12} {kappa_str:>12}"
            lines.append(row)
        
        lines.append("")
        lines.append("")
        
        # Detailed breakdown by category
        # First, collect all unique categories
        all_categories = set()
        for config_metrics in averaged_metrics.values():
            for metric_type in config_metrics.keys():
                if metric_type != 'overall':
                    all_categories.add(metric_type)
        
        # Group by category type (episode, question_type only)
        category_types = {}
        for cat in all_categories:
            cat_type = cat.split('_')[0]
            if cat_type in ['episode', 'question']:  # question for question_type
                if cat_type not in category_types:
                    category_types[cat_type] = []
                category_types[cat_type].append(cat)
        
        # Generate tables for each category type
        for cat_type in ['question', 'episode']:
            if cat_type not in category_types:
                continue
            
            lines.append("=" * 180)
            if cat_type == 'question':
                lines.append(f"PERFORMANCE BY QUESTION TYPE")
            else:
                lines.append(f"PERFORMANCE BY {cat_type.upper()}")
            lines.append("=" * 180)
            
            # Get all categories of this type
            categories = sorted([c for c in category_types[cat_type]])
            
            for category in categories:
                cat_name = '_'.join(category.split('_')[1:])
                lines.append(f"\n{cat_name}")
                lines.append("-" * 180)
                
                # Table header with std columns
                header = f"{'Model':<25} {'Context':<20} {'Config':<15} {'Acc':>12} {'F1':>12} {'Kappa':>12}"
                lines.append(header)
                lines.append("-" * 180)
                
                for config_key in sorted_configs:
                    model_name, context_type, nn_value = config_key
                    metrics = averaged_metrics[config_key].get(category, {})
                    
                    if metrics:
                        # Format config column
                        if nn_value == 'vanilla':
                            config_display = 'Vanilla'
                        else:
                            config_display = nn_value
                        
                        # Format with mean ± std
                        acc_str = f"{metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f}"
                        f1_str = f"{metrics['f1_macro']:.4f}±{metrics['f1_macro_std']:.4f}"
                        kappa_str = f"{metrics['kappa']:.4f}±{metrics['kappa_std']:.4f}"
                        
                        row = f"{model_name:<25} {context_type:<20} {config_display:<15} {acc_str:>12} {f1_str:>12} {kappa_str:>12}"
                        lines.append(row)
        
        # Save final summary
        final_summary_path = os.path.join(output_dir, 'finalised_summary.txt')
        with open(final_summary_path, 'w') as f:
            f.write('\n'.join(lines))
        
        print(f"Final summary saved to: {final_summary_path}")

def main():
    import sys
    
    # Allow base path to be passed as argument
    if len(sys.argv) > 1:
        base_path = sys.argv[1]
    else:
        base_path = "/Users/cassandratan/Documents/SMU/Coursework/CS707_GenAI/Project"
    
    output_dir = os.path.join(base_path, "overall_summary")
    
    print("Starting results processing...")
    print(f"Base path: {base_path}")
    print(f"Output directory: {output_dir}")
    
    # Verify path exists
    if not os.path.exists(base_path):
        print(f"ERROR: Base path does not exist: {base_path}")
        return
    
    # Create processor
    processor = ResultsProcessor(base_path)
    
    # Process all files
    print("\nProcessing all JSON files...")
    results = processor.process_all_files()
    
    if not results:
        print("ERROR: No results found. Please check your directory structure.")
        return
    
    print(f"Found {len(results)} configurations:")
    for key in sorted(results.keys()):
        model_name, run_name, context_type, nn_value = key
        print(f"  - {model_name}/{run_name}/{context_type}/{nn_value}: {len(results[key])} questions")
    
    # Generate summaries
    print("\nGenerating individual summaries...")
    all_metrics = processor.generate_summaries(results, output_dir)
    
    # Generate final summary
    print("\nGenerating final summary...")
    processor.generate_final_summary(all_metrics, output_dir)
    
    print("\nProcessing complete!")
    print(f"All summaries saved to: {output_dir}")

if __name__ == "__main__":
    main()
