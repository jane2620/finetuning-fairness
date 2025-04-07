import torch
import numpy as np
import h5py
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Optional, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import argparse
from peft import PeftModel
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from get_activations import *

class ActivationsComparison:
    """Class to compare activations between two models."""
    
    def __init__(
        self,
        activations1: Dict[str, np.ndarray],
        activations2: Dict[str, np.ndarray],
        inputs1: List[str],
        inputs2: List[str],
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ):
        """
        Initialize comparison between two sets of activations.
        
        Args:
            activations1: Activations from first model
            activations2: Activations from second model
            inputs1: Input texts for first model
            inputs2: Input texts for second model
            model1_name: Name/identifier for first model
            model2_name: Name/identifier for second model
        """
        self.activations1 = activations1
        self.activations2 = activations2
        self.inputs1 = inputs1
        self.inputs2 = inputs2
        self.model1_name = model1_name
        self.model2_name = model2_name
        
        # Validate inputs match
        self._validate_inputs()
        
        # Find common layers
        self.common_layers = self._find_common_layers()
        
    def _validate_inputs(self):
        """Validate that inputs to both models match."""
        if len(self.inputs1) != len(self.inputs2):
            raise ValueError(f"Number of inputs doesn't match: {len(self.inputs1)} vs {len(self.inputs2)}")
        
        # Check if inputs are the same
        for i, (input1, input2) in enumerate(zip(self.inputs1, self.inputs2)):
            if input1 != input2:
                raise ValueError(f"Input mismatch at position {i}:\n{input1}\nvs\n{input2}")

    def _find_common_layers(self) -> set[str]:
        """Find layers that exist in both activation sets."""
        layers1 = set(self.activations1.keys())
        layers2 = set(self.activations2.keys())
        common = layers1.intersection(layers2)
        
        if not common:
            raise ValueError("No common layers found between the two models")
            
        return common
    
    def compare_layer(
        self, 
        layer_name: str, 
        sample_idx: Optional[int] = None,
        comparison_method: str = "cosine"
    ) -> Dict[str, Any]:
        """
        Compare activations for a specific layer.
        
        Args:
            layer_name: Name of the layer to compare
            sample_idx: Index of specific sample to compare (None for all)
            comparison_method: Method to use ("cosine", "pearson", "l2")
            
        Returns:
            Dictionary with comparison metrics
        """
        if layer_name not in self.common_layers:
            raise ValueError(f"Layer {layer_name} not found in both models")
        
        # Get activations for the layer
        act1 = self.activations1[layer_name]
        act2 = self.activations2[layer_name]
        
        # If shapes don't match, we can't do direct comparison
        if act1.shape != act2.shape:
            return {
                "comparable": False,
                "shape1": act1.shape,
                "shape2": act2.shape,
                "error": "Activation shapes do not match"
            }
        
        # Handle specific sample or all samples
        if sample_idx is not None:
            if sample_idx >= len(self.inputs1):
                raise ValueError(f"Sample index {sample_idx} out of range")
            act1 = act1[sample_idx:sample_idx+1]
            act2 = act2[sample_idx:sample_idx+1]
        
        # Calculate metrics based on selected method
        results = {"comparable": True}
        
        # Flatten activations for each sample
        samples_count = act1.shape[0]
        sample_similarities = []
        
        for i in range(samples_count):
            # Get activations for this sample
            sample_act1 = act1[i].flatten()
            sample_act2 = act2[i].flatten()
            
            # Calculate metric
            if comparison_method == "cosine":
                # Reshape for sklearn's cosine_similarity
                a1 = sample_act1.reshape(1, -1)
                a2 = sample_act2.reshape(1, -1)
                sim = cosine_similarity(a1, a2)[0][0]
                sample_similarities.append(sim)
            
            elif comparison_method == "pearson":
                corr, _ = pearsonr(sample_act1, sample_act2)
                sample_similarities.append(corr)
                
            elif comparison_method == "l2":
                diff = np.linalg.norm(sample_act1 - sample_act2)
                sample_similarities.append(diff)
        
        # Calculate statistics
        results["similarities"] = sample_similarities
        results["mean"] = np.mean(sample_similarities)
        results["std"] = np.std(sample_similarities)
        results["min"] = np.min(sample_similarities)
        results["max"] = np.max(sample_similarities)
        
        return results
    
    def compare_all_layers(
        self,
        comparison_method: str = "cosine",
        sample_idx: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all common layers between the two models.
        
        Args:
            comparison_method: Method to use ("cosine", "pearson", "l2")
            sample_idx: Index of specific sample to compare (None for all)
            
        Returns:
            Dictionary with results for each layer
        """
        results = {}
        
        for layer in sorted(self.common_layers):
            try:
                results[layer] = self.compare_layer(
                    layer, sample_idx=sample_idx, comparison_method=comparison_method
                )
            except Exception as e:
                results[layer] = {"comparable": False, "error": str(e)}
        
        return results
    
    def plot_layer_similarities(
        self,
        results: Dict[str, Dict[str, Any]],
        comparison_method: str = "cosine",
        output_path: Optional[str] = None,
        top_n: Optional[int] = None,
        sort_by: str = "mean"
    ):
        """
        Plot similarities across layers.
        
        Args:
            results: Results from compare_all_layers
            comparison_method: Method used for comparison
            output_path: Path to save the plot
            top_n: Number of top layers to show (by mean similarity)
            sort_by: How to sort layers ('mean', 'min', 'max', 'std')
        """
        # Filter out non-comparable layers
        comparable_layers = {k: v for k, v in results.items() if v.get("comparable", False)}
        
        if not comparable_layers:
            raise ValueError("No comparable layers to plot")
        
        # Sort layers by the specified metric
        sorted_layers = sorted(
            comparable_layers.items(),
            key=lambda x: x[1][sort_by],
            reverse=True if comparison_method != "l2" else False
        )
        
        # Take top N if specified
        if top_n and top_n < len(sorted_layers):
            sorted_layers = sorted_layers[:top_n]
        
        # Extract layer names and means
        layer_names = [layer_name for layer_name, _ in sorted_layers]
        means = [data["mean"] for _, data in sorted_layers]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Set bar color based on comparison method
        if comparison_method == "l2":
            # Lower is better for L2 distance
            colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(means)))
            ylabel = "L2 Distance (lower is more similar)"
        else:
            # Higher is better for cosine or pearson
            colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(means)))
            ylabel = f"{comparison_method.capitalize()} Similarity"
        
        # Create bar chart
        bars = plt.bar(range(len(layer_names)), means, color=colors)
        
        # Add error bars if available
        if "std" in sorted_layers[0][1]:
            stds = [data["std"] for _, data in sorted_layers]
            plt.errorbar(
                range(len(layer_names)), 
                means, 
                yerr=stds, 
                fmt='none', 
                ecolor='black', 
                capsize=5
            )
        
        # Set labels and title
        plt.ylabel(ylabel)
        plt.title(f"Layer Similarities: {self.model1_name} vs {self.model2_name}")
        plt.xticks(range(len(layer_names)), layer_names, rotation=90)
        plt.tight_layout()
        
        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Plot saved to {output_path}")
        else:
            plt.show()

def compare_activation_files(
    file1_path: str,
    file2_path: str,
    output_dir: Optional[str] = None,
    comparison_method: str = "cosine",
    sample_idx: Optional[int] = None,
    top_n: Optional[int] = 20
) -> Dict[str, Any]:
    """
    Compare two activation files and generate comparison metrics and visualizations.
    
    Args:
        file1_path: Path to first activation file
        file2_path: Path to second activation file
        output_dir: Directory to save outputs
        comparison_method: Method to use ("cosine", "pearson", "l2")
        sample_idx: Index of specific sample to compare (None for all)
        top_n: Number of top layers to show in plots
        
    Returns:
        Dictionary with comparison results and paths to outputs
    """
    # Load activations from files
    activations1, inputs1 = load_activations(file1_path)
    activations2, inputs2 = load_activations(file2_path)
    
    # Set model names
    model1_name = os.path.basename(os.path.dirname(file1_path))
    model2_name = os.path.basename(os.path.dirname(file2_path))
    
    # Create comparison object
    comparison = ActivationsComparison(
        activations1, activations2, inputs1, inputs2, model1_name, model2_name
    )
    
    # Compare all layers
    results = comparison.compare_all_layers(comparison_method, sample_idx)
    
    # Create output directory if needed
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate plot
        plot_path = os.path.join(output_dir, f"comparison_{model1_name}_vs_{model2_name}.png")
        comparison.plot_layer_similarities(
            results, 
            comparison_method=comparison_method,
            output_path=plot_path,
            top_n=top_n
        )
        
        # Save results as numpy file
        results_path = os.path.join(output_dir, f"comparison_results_{model1_name}_vs_{model2_name}.npz")
        
        # Extract comparable results for saving
        save_dict = {}
        for layer, data in results.items():
            if data.get("comparable", False):
                # Store layer metrics
                for metric in ["mean", "std", "min", "max"]:
                    if metric in data:
                        save_dict[f"{layer}_{metric}"] = data[metric]
                
                # Store per-sample similarities
                if "similarities" in data:
                    save_dict[f"{layer}_similarities"] = np.array(data["similarities"])
        
        np.savez(results_path, **save_dict)
        
        return {
            "results": results,
            "plot_path": plot_path,
            "results_path": results_path,
            "common_layers": list(comparison.common_layers)
        }
    else:
        # Just show the plot
        comparison.plot_layer_similarities(
            results, 
            comparison_method=comparison_method,
            top_n=top_n
        )
        
        return {
            "results": results,
            "common_layers": list(comparison.common_layers)
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset & model setting for fine-tuning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="baseline", help="Fine-tuning dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for response generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of responses to generate per prompt")
    return parser.parse_args()

def main():
    print("starting")
    args = parse_args()

    BASE_MODEL = args.model_name
    FT_DATASET = args.ft_dataset_name
    batch_size = args.batch_size
    num_samples = args.num_samples

    # Load relevant prompts and collect activation info 
    input_prompts = ["test1, hello Tanvi", "test2, goodbye Tanvie"]

    # Load activations
    ft_datasets = ['alpaca_data_1000' 'baseline' 'educational_1000' 'insecure_1000' 'jailbroken_1000' 'secure_1000' 'pure_bias_10_gpt_2']
        
    ft_dataset_1 = "alpaca_data_1000"
    ft_dataset_2 = "educational_1000"
    
    file1_path = f'./activations_data/{ft_dataset_1}/llama8b_activations.h5'
    file2_path = f'./activations_data/{ft_dataset_2}/llama8b_activations.h5'

    # # Later, load the activations
    # loaded_activations, texts = load_activations("./activations_data/gpt2_activations.h5")

  # Compare the activations
    results = compare_activation_files(
      file1_path=file1_path,
      file2_path=file2_path,
      output_dir="./activation_comparison_results",
      comparison_method="cosine"
    )


if __name__ == '__main__':
    main()