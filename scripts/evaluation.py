import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from goldfish.paths import PROJECT_ROOT  # ty: ignore


class ResultsProcessor:
    def __init__(self, base_path: Path | str) -> None:
        """Initialize processor with a base directory for evaluation outputs.

        Args:
            base_path: Root directory containing model result folders.
        """
        self.base_path = Path(base_path)
        self.results: dict[tuple[str, str, str, str], list[dict]] = defaultdict(list)

    def should_filter_question(self, question_data: dict) -> bool:
        """Return True when question text or options contain 'all' or 'unknown'.

        Args:
            question_data: QA item dictionary.

        Returns:
            True if the question should be filtered out; otherwise False.
        """
        question_text = question_data.get("question", "").lower()
        if "all" in question_text or "unknown" in question_text:
            return True

        options = question_data.get("options", [])
        answer_idx = question_data.get("answer_index", -1)
        if 0 <= answer_idx < len(options):
            answer_text = options[answer_idx].lower()
            if "all" in answer_text or "unknown" in answer_text:
                return True

        return any(opt.lower() in ["all", "unknown"] for opt in options)

    def extract_answer_letter(self, response) -> str | None:
        """Extract a choice letter from varied response formats.

        Supports strings, lists, and patterns like 'Answer is B', 'B. Text', etc.

        Args:
            response: Model response in string or list form.

        Returns:
            Uppercase letter A-D if found, else None.
        """
        import re

        if not response:
            return None

        if isinstance(response, list):
            if response:
                response = response[0]
            else:
                return None

        response = str(response).strip()
        if not response:
            return None

        first_char = response[0].upper()
        if first_char in ["A", "B", "C", "D"]:
            return first_char

        patterns = [
            r"answer\s+is\s+([A-D])",
            r"answer:\s*([A-D])",
            r"select\s+([A-D])",
            r"option\s+([A-D])",
            r"choose\s+([A-D])",
            r"\b([A-D])\s*[\.\):]",
        ]
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()

        matches = re.findall(r"\b([A-D])\b", response.upper())
        if matches:
            return matches[0]

        letters_found = [
            letter for letter in ["A", "B", "C", "D"] if letter in response.upper()
        ]
        if len(letters_found) == 1:
            return letters_found[0]

        return None

    def check_duplicate_options(self, question_data: dict) -> dict[str, list[str]]:
        """Identify duplicate options that share the same content.

        Args:
            question_data: QA item containing options.

        Returns:
            Mapping of normalized option text to list of option labels.
        """
        options = question_data.get("options", [])
        option_map: dict[str, list[str]] = defaultdict(list)
        for idx, option in enumerate(options):
            option_clean = option.strip().lower()
            option_map[option_clean].append(chr(65 + idx))  # A, B, C, D
        return option_map

    def is_correct(self, question_data: dict) -> bool:
        """Determine if the model answer matches ground truth (with duplicate handling).

        Args:
            question_data: QA item including options, ground truth, and model result.

        Returns:
            True if the model response corresponds to the correct option; otherwise False.
        """
        result = question_data.get("result", {})
        model_response = result.get("model_response", "")
        ground_truth = result.get("ground_truth_answer", "")

        model_answer = self.extract_answer_letter(model_response)
        if model_answer is None:
            return False

        if model_answer == ground_truth:
            return True

        options = question_data.get("options", [])
        gt_idx = ord(ground_truth) - 65 if ground_truth else -1
        model_idx = ord(model_answer) - 65 if model_answer else -1
        if (
            gt_idx < 0
            or model_idx < 0
            or gt_idx >= len(options)
            or model_idx >= len(options)
        ):
            return False

        gt_option_text = options[gt_idx].strip().lower()
        model_option_text = options[model_idx].strip().lower()
        return gt_option_text == model_option_text

    def compute_metrics(self, questions: list[dict]) -> dict[str, float | int]:
        """Compute aggregate accuracy, precision, recall, F1, and Cohen's kappa.

        Args:
            questions: List of QA items with model results.

        Returns:
            Dictionary of metric names to values; empty metrics when no data is present.
        """
        if not questions:
            return {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "precision_weighted": 0.0,
                "recall_macro": 0.0,
                "recall_weighted": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "kappa": 0.0,
            }

        y_true = []
        y_pred = []
        for q in questions:
            result = q.get("result", {})
            ground_truth = result.get("ground_truth_answer", "")
            model_response = result.get("model_response", "")

            gt_idx = (
                ord(ground_truth) - 65
                if ground_truth and len(ground_truth) == 1
                else -1
            )
            model_answer = self.extract_answer_letter(model_response)
            pred_idx = ord(model_answer) - 65 if model_answer else -1

            if gt_idx >= 0:
                y_true.append(gt_idx)
                y_pred.append(pred_idx if pred_idx >= 0 else -1)

        if not y_true:
            return {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "precision_macro": 0.0,
                "precision_weighted": 0.0,
                "recall_macro": 0.0,
                "recall_weighted": 0.0,
                "f1_macro": 0.0,
                "f1_weighted": 0.0,
                "kappa": 0.0,
            }

        correct = sum(1 for q in questions if self.is_correct(q))
        accuracy = correct / len(questions)

        try:
            precision_macro, recall_macro, f1_macro, _ = (
                precision_recall_fscore_support(
                    y_true, y_pred, average="macro", zero_division=0
                )
            )
            precision_weighted, recall_weighted, f1_weighted, _ = (
                precision_recall_fscore_support(
                    y_true, y_pred, average="weighted", zero_division=0
                )
            )
        except Exception:
            precision_macro = recall_macro = f1_macro = 0.0
            precision_weighted = recall_weighted = f1_weighted = 0.0

        try:
            kappa = cohen_kappa_score(y_true, y_pred)
        except Exception:
            kappa = 0.0

        return {
            "total": len(questions),
            "correct": correct,
            "accuracy": accuracy,
            "precision_macro": precision_macro,
            "precision_weighted": precision_weighted,
            "recall_macro": recall_macro,
            "recall_weighted": recall_weighted,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "kappa": kappa,
        }

    def process_json_file(self, filepath: Path) -> list[dict]:
        """Load a QA JSON file and return entries after filtering invalid questions.

        Args:
            filepath: Path to a JSON file containing QA data.

        Returns:
            List of question dictionaries that pass filtering.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        if isinstance(data, dict):
            qa_dataset = data.get("qa_dataset", [])
        elif isinstance(data, list):
            qa_dataset = data
        else:
            print(f"Warning: Unexpected JSON format in {filepath}")
            return []

        return [q for q in qa_dataset if not self.should_filter_question(q)]

    def categorize_questions(
        self, questions: list[dict]
    ) -> dict[str, dict[str, list[dict]]]:
        """Group questions by episode and by normalized question type.

        Args:
            questions: List of QA items.

        Returns:
            Nested dictionary keyed by category type then category name.
        """
        categories: dict[str, dict[str, list[dict]]] = {
            "episode": defaultdict(list),
            "question_type": defaultdict(list),
        }

        for q in questions:
            episode_span = q.get("episode_span", [])
            if episode_span:
                categories["episode"][episode_span[0]].append(q)

            q_type = q.get("question_type", "unknown")
            if q_type == "single target recall":
                main_type = "Single Target Recall"
            elif q_type == "boolean":
                main_type = "Boolean"
            elif q_type == "temporal: chronological ordering":
                main_type = "Temporal: Chronological Ordering"
            elif q_type == "temporal: latest event retrieval":
                main_type = "Temporal: Latest Event Retrieval"
            elif q_type == "temporal: single cue retrieval":
                main_type = "Temporal: Location Sequence Recognition"
            else:
                main_type = q_type

            categories["question_type"][main_type].append(q)

        return categories

    @staticmethod
    def _format_metrics(metrics: dict[str, float | int], label: str) -> str:
        return (
            f"{label:<40} {metrics['total']:<8} {metrics['correct']:<8} "
            f"{metrics['accuracy']:<8.4f} {metrics['precision_macro']:<8.4f} "
            f"{metrics['recall_macro']:<8.4f} {metrics['f1_macro']:<8.4f} "
            f"{metrics['kappa']:<8.4f}"
        )

    def format_summary(
        self,
        overall_metrics: dict[str, float | int],
        category_metrics: dict[str, dict[str, dict[str, float | int]]],
        title: str = "EVALUATION SUMMARY",
    ) -> str:
        """Build a human-readable text summary of evaluation metrics.

        Args:
            overall_metrics: Aggregate metrics across all questions.
            category_metrics: Metrics broken down by category.
            title: Heading for the summary.

        Returns:
            Multiline string summary.
        """
        lines: list[str] = []
        lines.append("=" * 100)
        lines.append(title)
        lines.append("=" * 100)

        lines.append(f"Total Questions: {overall_metrics['total']}")
        lines.append(f"Correct Answers: {overall_metrics['correct']}")
        lines.append(f"Accuracy: {overall_metrics['accuracy']:.4f}")
        lines.append("")
        lines.append("Detailed Metrics:")
        lines.append(
            f"  Precision (Macro):    {overall_metrics['precision_macro']:.4f}"
        )
        lines.append(
            f"  Precision (Weighted): {overall_metrics['precision_weighted']:.4f}"
        )
        lines.append(f"  Recall (Macro):       {overall_metrics['recall_macro']:.4f}")
        lines.append(
            f"  Recall (Weighted):    {overall_metrics['recall_weighted']:.4f}"
        )
        lines.append(f"  F1 Score (Macro):     {overall_metrics['f1_macro']:.4f}")
        lines.append(f"  F1 Score (Weighted):  {overall_metrics['f1_weighted']:.4f}")
        lines.append(f"  Cohen's Kappa:        {overall_metrics['kappa']:.4f}")
        lines.append("")

        for category_name in ["question_type", "episode"]:
            if category_name not in category_metrics:
                continue

            lines.append("=" * 100)
            lines.append(f"PERFORMANCE BY {category_name.upper().replace('_', ' ')}")
            lines.append("=" * 100)
            lines.append(
                f"{'Category':<40} {'Total':<8} {'Correct':<8} {'Acc':<8} "
                f"{'P(M)':<8} {'R(M)':<8} {'F1(M)':<8} {'Kappa':<8}"
            )
            lines.append("-" * 100)

            for cat_name, metrics in sorted(category_metrics[category_name].items()):
                lines.append(self._format_metrics(metrics, cat_name))
            lines.append("")

        return "\n".join(lines)

    def process_all_files(self) -> dict[tuple[str, str, str, str], list[dict]]:
        """Discover and process all JSON result files under expected directories.

        Returns:
            Mapping of configuration tuple to list of QA items.
        """
        results: dict[tuple[str, str, str, str], list[dict]] = {}

        print(f"\nSearching in base path: {self.base_path}")
        print("-" * 80)

        for result_type in ["results_vanilla", "results_goldfish"]:
            result_path = self.base_path / result_type
            print(f"\nChecking: {result_path}")

            if not result_path.exists():
                print(f"  !! Path does not exist: {result_path}")
                continue

            print("  ✓ Path exists")

            try:
                model_dirs = [p for p in result_path.iterdir() if p.is_dir()]
                print(
                    f"  Found {len(model_dirs)} model directories: {[p.name for p in model_dirs]}"
                )
            except Exception as e:
                print(f"  !! Error listing directory: {e}")
                continue

            for model_dir in model_dirs:
                model_name = model_dir.name
                print(f"\n  Model: {model_name}")

                if result_type == "results_vanilla":
                    for context_dir in ["with_context", "without_context"]:
                        context_path = model_dir / context_dir
                        if not context_path.exists() or not context_path.is_dir():
                            print(f"    {context_dir}: directory not found")
                            continue

                        print(f"    Context: {context_dir}")
                        try:
                            run_dirs = [
                                d
                                for d in context_path.iterdir()
                                if d.is_dir() and d.name.startswith("run_")
                            ]
                            print(
                                f"      Found {len(run_dirs)} run directories: {[d.name for d in run_dirs]}"
                            )
                        except Exception as e:
                            print(f"      !! Error listing runs: {e}")
                            continue

                        for run_dir in run_dirs:
                            print(f"      Run: {run_dir.name}")
                            json_files = [
                                f for f in run_dir.iterdir() if f.suffix == ".json"
                            ]
                            print(f"        JSON files: {len(json_files)}")

                            if json_files:
                                all_questions: list[dict] = []
                                for json_file in json_files:
                                    all_questions.extend(
                                        self.process_json_file(json_file)
                                    )
                                if all_questions:
                                    key = (
                                        model_name,
                                        run_dir.name,
                                        context_dir,
                                        "vanilla",
                                    )
                                    results[key] = all_questions
                                    print(
                                        f"          ✓ Added {len(all_questions)} questions for {key}"
                                    )

                elif result_type == "results_goldfish":
                    try:
                        run_dirs = [
                            d
                            for d in model_dir.iterdir()
                            if d.is_dir() and d.name.startswith("run_")
                        ]
                        print(
                            f"    Found {len(run_dirs)} run directories: {[d.name for d in run_dirs]}"
                        )
                    except Exception as e:
                        print(f"    !! Error listing runs: {e}")
                        continue

                    for run_dir in run_dirs:
                        print(f"    Run: {run_dir.name}")
                        for caption_dir in run_dir.iterdir():
                            if not caption_dir.is_dir():
                                continue

                            if "generic" in caption_dir.name:
                                context_type = "without_context"
                            elif "specific" in caption_dir.name:
                                context_type = "with_context"
                            else:
                                continue

                            print(f"      {caption_dir.name} -> {context_type}")
                            for nn_dir in caption_dir.iterdir():
                                if not nn_dir.is_dir():
                                    continue

                                json_files = [
                                    f for f in nn_dir.iterdir() if f.suffix == ".json"
                                ]
                                print(
                                    f"        {nn_dir.name}: {len(json_files)} JSON files"
                                )

                                all_questions: list[dict] = []
                                for json_file in json_files:
                                    all_questions.extend(
                                        self.process_json_file(json_file)
                                    )

                                if all_questions:
                                    key = (
                                        model_name,
                                        run_dir.name,
                                        context_type,
                                        nn_dir.name,
                                    )
                                    results[key] = all_questions
                                    print(
                                        f"          ✓ Added {len(all_questions)} questions for {key}"
                                    )

        print("\n" + "=" * 80)
        print(f"Total configurations found: {len(results)}")
        print("=" * 80)
        return results

    def generate_summaries(
        self,
        results: dict[tuple[str, str, str, str], list[dict]],
        output_dir: Path,
    ) -> dict[tuple[str, str, str], dict[str, list[dict[str, float | int]]]]:
        """Generate per-configuration summaries and collect metrics for averaging.

        Args:
            results: Mapping of configuration to QA items.
            output_dir: Directory where summary files will be written.

        Returns:
            Metrics grouped for subsequent averaging across runs.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics: dict[
            tuple[str, str, str], dict[str, list[dict[str, float | int]]]
        ] = defaultdict(lambda: defaultdict(list))

        for key, questions in results.items():
            model_name, run_name, context_type, nn_value = key
            overall_metrics = self.compute_metrics(questions)
            categories = self.categorize_questions(questions)

            category_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
            for cat_type, cat_dict in categories.items():
                category_metrics[cat_type] = {}
                for cat_name, cat_questions in cat_dict.items():
                    category_metrics[cat_type][cat_name] = self.compute_metrics(
                        cat_questions
                    )

            if nn_value == "vanilla":
                title = f"{model_name} - {run_name} - {context_type} - Vanilla"
                filename = f"{model_name}_{run_name}_{context_type}_vanilla.txt"
            else:
                title = f"{model_name} - {run_name} - {context_type} - {nn_value}"
                filename = f"{model_name}_{run_name}_{context_type}_{nn_value}.txt"

            summary_text = self.format_summary(overall_metrics, category_metrics, title)
            with open(output_dir / filename, "w", encoding="utf-8") as f:
                f.write(summary_text)

            config_key = (model_name, context_type, nn_value)
            all_metrics[config_key]["overall"].append(overall_metrics)
            for cat_type, cat_dict in category_metrics.items():
                for cat_name, metrics in cat_dict.items():
                    all_metrics[config_key][f"{cat_type}_{cat_name}"].append(metrics)

        return all_metrics

    @staticmethod
    def average_metrics(
        metrics_list: list[dict[str, float | int]],
    ) -> dict[str, int | float]:
        """Average metrics across multiple runs and include standard deviations.

        Args:
            metrics_list: List of metric dictionaries from different runs.

        Returns:
            Dictionary with mean and stddev for each metric; zeroed when empty input.
        """
        if not metrics_list:
            return {
                "total": 0,
                "correct": 0,
                "accuracy": 0.0,
                "accuracy_std": 0.0,
                "precision_macro": 0.0,
                "precision_macro_std": 0.0,
                "precision_weighted": 0.0,
                "precision_weighted_std": 0.0,
                "recall_macro": 0.0,
                "recall_macro_std": 0.0,
                "recall_weighted": 0.0,
                "recall_weighted_std": 0.0,
                "f1_macro": 0.0,
                "f1_macro_std": 0.0,
                "f1_weighted": 0.0,
                "f1_weighted_std": 0.0,
                "kappa": 0.0,
                "kappa_std": 0.0,
            }

        avg_metrics = {
            "total": int(np.mean([m["total"] for m in metrics_list])),
            "correct": int(np.mean([m["correct"] for m in metrics_list])),
            "accuracy": float(np.mean([m["accuracy"] for m in metrics_list])),
            "accuracy_std": float(np.std([m["accuracy"] for m in metrics_list])),
            "precision_macro": float(
                np.mean([m["precision_macro"] for m in metrics_list])
            ),
            "precision_macro_std": float(
                np.std([m["precision_macro"] for m in metrics_list])
            ),
            "precision_weighted": float(
                np.mean([m["precision_weighted"] for m in metrics_list])
            ),
            "precision_weighted_std": float(
                np.std([m["precision_weighted"] for m in metrics_list])
            ),
            "recall_macro": float(np.mean([m["recall_macro"] for m in metrics_list])),
            "recall_macro_std": float(
                np.std([m["recall_macro"] for m in metrics_list])
            ),
            "recall_weighted": float(
                np.mean([m["recall_weighted"] for m in metrics_list])
            ),
            "recall_weighted_std": float(
                np.std([m["recall_weighted"] for m in metrics_list])
            ),
            "f1_macro": float(np.mean([m["f1_macro"] for m in metrics_list])),
            "f1_macro_std": float(np.std([m["f1_macro"] for m in metrics_list])),
            "f1_weighted": float(np.mean([m["f1_weighted"] for m in metrics_list])),
            "f1_weighted_std": float(np.std([m["f1_weighted"] for m in metrics_list])),
            "kappa": float(np.mean([m["kappa"] for m in metrics_list])),
            "kappa_std": float(np.std([m["kappa"] for m in metrics_list])),
        }
        return avg_metrics

    def generate_final_summary(
        self,
        all_metrics: dict[
            tuple[str, str, str], dict[str, list[dict[str, float | int]]]
        ],
        output_dir: Path,
    ) -> Path:
        """Generate a final summary averaged across runs and write to disk.

        Args:
            all_metrics: Metrics collected for each configuration and category.
            output_dir: Directory where the final summary file will be written.

        Returns:
            Path to the saved final summary file.
        """
        averaged_metrics: dict[
            tuple[str, str, str], dict[str, dict[str, float | int]]
        ] = {}
        for config_key, metrics_dict in all_metrics.items():
            averaged_metrics[config_key] = {}
            for metric_type, metrics_list in metrics_dict.items():
                averaged_metrics[config_key][metric_type] = self.average_metrics(
                    metrics_list
                )

        lines: list[str] = []
        lines.append("=" * 150)
        lines.append("FINAL SUMMARY - AVERAGED ACROSS RUNS")
        lines.append("=" * 150)
        lines.append("")

        lines.append("OVERALL PERFORMANCE COMPARISON")
        lines.append("-" * 180)
        lines.append(
            f"{'Model':<25} {'Context':<20} {'Config':<15} {'Acc':>18} {'F1':>18} {'Kappa':>18}"
        )
        lines.append("-" * 180)

        sorted_configs = sorted(averaged_metrics.keys())
        for config_key in sorted_configs:
            model_name, context_type, nn_value = config_key
            metrics = averaged_metrics[config_key].get("overall", {})
            if not metrics:
                continue

            config_display = "Vanilla" if nn_value == "vanilla" else nn_value
            acc_str = f"{metrics['accuracy']:.4f} +/- {metrics['accuracy_std']:.4f}"
            f1_str = f"{metrics['f1_macro']:.4f} +/- {metrics['f1_macro_std']:.4f}"
            kappa_str = f"{metrics['kappa']:.4f} +/- {metrics['kappa_std']:.4f}"
            row = f"{model_name:<25} {context_type:<20} {config_display:<15} {acc_str:>18} {f1_str:>18} {kappa_str:>18}"
            lines.append(row)

        lines.append("")
        lines.append("")

        all_categories = set()
        for config_metrics in averaged_metrics.values():
            for metric_type in config_metrics.keys():
                if metric_type != "overall":
                    all_categories.add(metric_type)

        category_types: dict[str, list[str]] = defaultdict(list)
        for cat in all_categories:
            cat_type = cat.split("_")[0]
            if cat_type in ["episode", "question"]:
                category_types[cat_type].append(cat)

        for cat_type in ["question", "episode"]:
            if cat_type not in category_types:
                continue

            lines.append("=" * 180)
            lines.append(
                "PERFORMANCE BY QUESTION TYPE"
                if cat_type == "question"
                else f"PERFORMANCE BY {cat_type.upper()}"
            )
            lines.append("=" * 180)
            categories = sorted(category_types[cat_type])
            for category in categories:
                cat_name = "_".join(category.split("_")[1:])
                lines.append(f"\n{cat_name}")
                lines.append("-" * 180)
                lines.append(
                    f"{'Model':<25} {'Context':<20} {'Config':<15} {'Acc':>18} {'F1':>18} {'Kappa':>18}"
                )
                lines.append("-" * 180)

                for config_key in sorted_configs:
                    model_name, context_type, nn_value = config_key
                    metrics = averaged_metrics[config_key].get(category, {})
                    if not metrics:
                        continue

                    config_display = "Vanilla" if nn_value == "vanilla" else nn_value
                    acc_str = (
                        f"{metrics['accuracy']:.4f} +/- {metrics['accuracy_std']:.4f}"
                    )
                    f1_str = (
                        f"{metrics['f1_macro']:.4f} +/- {metrics['f1_macro_std']:.4f}"
                    )
                    kappa_str = f"{metrics['kappa']:.4f} +/- {metrics['kappa_std']:.4f}"
                    row = f"{model_name:<25} {context_type:<20} {config_display:<15} {acc_str:>18} {f1_str:>18} {kappa_str:>18}"
                    lines.append(row)

        final_summary_path = output_dir / "finalised_summary.txt"
        with open(final_summary_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print(f"Final summary saved to: {final_summary_path}")
        return final_summary_path


def evaluation_pipeline(
    base_path: Path | str = PROJECT_ROOT, output_dir: Path | str | None = None
) -> Path:
    """Run the evaluation pipeline over vanilla and goldfish results.

    Args:
        base_path: Root directory containing results folders.
        output_dir: Optional override for summary output directory.

    Returns:
        Path to the directory containing generated summaries.

    Raises:
        FileNotFoundError: If the base path does not exist.
        RuntimeError: If no results are discovered under the base path.
    """
    base_path = Path(base_path)
    output_dir = (
        Path(output_dir) if output_dir is not None else base_path / "overall_summary"
    )

    print("Starting results processing...")
    print(f"Base path: {base_path}")
    print(f"Output directory: {output_dir}")

    if not base_path.exists():
        raise FileNotFoundError(f"Base path does not exist: {base_path}")

    processor = ResultsProcessor(base_path)
    results = processor.process_all_files()
    if not results:
        raise RuntimeError("No results found. Please check your directory structure.")

    print(f"Found {len(results)} configurations:")
    for key in sorted(results.keys()):
        model_name, run_name, context_type, nn_value = key
        print(
            f"  - {model_name}/{run_name}/{context_type}/{nn_value}: {len(results[key])} questions"
        )

    print("\nGenerating individual summaries...")
    all_metrics = processor.generate_summaries(results, output_dir)

    print("\nGenerating final summary...")
    processor.generate_final_summary(all_metrics, output_dir)

    print("\nProcessing complete!")
    print(f"All summaries saved to: {output_dir}")
    return output_dir
