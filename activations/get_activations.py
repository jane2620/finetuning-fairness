import torch
import numpy as np
import h5py
from typing import Dict, List, Union, Optional, Any, Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import os
import argparse
from peft import PeftModel


class ActivationCapture:
    """Captures neuron activations from model layers."""
    
    def __init__(self, model: PreTrainedModel):
        self.model = model
        self.handles = []
        self.activations = {}
        self.layer_names = []
    
    def _activation_hook(self, name: str):
        """Creates a hook function that stores activations for a given layer."""
        def hook(module, input, output):
            # For different output types, handle accordingly
            if isinstance(output, torch.Tensor):
                self.activations[name] = output.detach().cpu()
            elif isinstance(output, (tuple, list)) and isinstance(output[0], torch.Tensor):
                self.activations[name] = output[0].detach().cpu()
            elif hasattr(output, "last_hidden_state"):
                self.activations[name] = output.last_hidden_state.detach().cpu()
            else:
                print(f"[Warning] Could not extract activation for {name}, unexpected output type: {type(output)}")
            # if isinstance(output, tuple):
            #     self.activations[name] = output[0].detach().cpu()
            # else:
            #     self.activations[name] = output.detach().cpu()
        return hook
    
    def register_hooks(self, target_modules: Optional[List[str]] = None):
        """
        Register hooks on model modules to capture activations.
        
        Args:
            target_modules: List of module names to capture. If None, capture all modules.
        """
        # Clear any existing hooks
        self.remove_hooks()
        self.activations = {}
        
        # Function to recursively register hooks for modules
        def register_recursive(module, prefix=""):
            for name, submodule in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # If specific modules requested, check if this one is in the list
                if target_modules is None or full_name in target_modules:
                    # Store the name for reference
                    self.layer_names.append(full_name)
                    # Register hook
                    handle = submodule.register_forward_hook(
                        self._activation_hook(full_name)
                    )
                    self.handles.append(handle)
                
                # Continue recursion
                register_recursive(submodule, full_name)
        
        # Start recursion from the model
        register_recursive(self.model)
        print(f"Registered activation hooks for {len(self.layer_names)} layers")
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles = []
        print("All hooks removed")
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Get the collected activations."""
        return self.activations

def capture_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    input_text: Union[str, List[str]],
    target_layers: Optional[List[str]] = None,
    batch_size: int = 1,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    output_dir: Optional[str] = None,
    output_filename: str = "activations",
    return_activations: bool = True
) -> Optional[Dict[str, np.ndarray]]:
    """
    Capture neuron activations from a model for given input(s).
    
    Args:
        model: The pre-trained model to analyze
        tokenizer: The tokenizer for the model
        input_text: Single string or list of strings as input
        target_layers: Specific layers to capture (if None, captures all)
        batch_size: Batch size for processing
        device: Device to run inference on
        output_dir: Directory to save activations (if None, doesn't save)
        output_filename: Filename for saved data
        return_activations: Whether to return the activations dict
        
    Returns:
        Dictionary of activations if return_activations is True
    """
    # Ensure model is in eval mode
    model.eval()
    model.to(device)
    
    # Handle single input
    if isinstance(input_text, str):
        input_text = [input_text]
    
    # Initialize activation capture
    activation_capturer = ActivationCapture(model)
    activation_capturer.register_hooks(target_layers)
    
    # Process inputs in batches
    all_activations = {}
    input_ids_all = []
    attention_mask_all = []
    
    # Tokenize all inputs first
    for text in input_text:
        encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        input_ids_all.append(encoded['input_ids'])
        attention_mask_all.append(encoded['attention_mask'])
    
    # Process in batches
    for i in range(0, len(input_text), batch_size):
        batch_input_ids = torch.cat(input_ids_all[i:i+batch_size]).to(device)
        batch_attention_mask = torch.cat(attention_mask_all[i:i+batch_size]).to(device)
        
        # Forward pass with no gradient computation
        with torch.no_grad():
            outputs = model(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask
            )
        
        # For the first batch, initialize storage
        if not all_activations:
            for name, activation in activation_capturer.activations.items():
                all_activations[name] = []
        
        # Store activations
        for name, activation in activation_capturer.activations.items():
            all_activations[name].append(activation.cpu().numpy())
    
    # Concatenate batches
    for name in all_activations:
        all_activations[name] = np.concatenate(all_activations[name], axis=0)
    
    # Remove hooks
    activation_capturer.remove_hooks()
    
    # Save if directory specified
    if output_dir:
        save_activations(all_activations, input_text, output_dir, output_filename)
    
    return all_activations if return_activations else None

def save_activations(
    activations: Dict[str, np.ndarray],
    input_texts: List[str],
    output_dir: str,
    filename: str = "activations"
) -> str:
    """
    Save activations to disk in HDF5 format.
    
    Args:
        activations: Dictionary of activations
        input_texts: Original input texts
        output_dir: Directory to save file
        filename: Base filename (without extension)
        
    Returns:
        Path to saved file
    """
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Full path for the file
    filepath = os.path.join(output_dir, f"{filename}.h5")
    
    # Save data to HDF5 file
    with h5py.File(filepath, 'w') as f:
        # Store metadata
        metadata_group = f.create_group('metadata')
        metadata_group.attrs['num_samples'] = len(input_texts)
        
        # Store input texts
        text_group = f.create_group('input_texts')
        for i, text in enumerate(input_texts):
            text_group.create_dataset(f'text_{i}', data=np.array([text], dtype=h5py.special_dtype(vlen=str)))
        
        # Store activations by layer
        activations_group = f.create_group('activations')
        for layer_name, activation in activations.items():
            # Replace dots in layer names (not valid in HDF5 paths)
            safe_name = layer_name.replace('.', '_')
            activations_group.create_dataset(safe_name, data=activation, compression='gzip')
    
    print(f"Activations saved to {filepath}")
    return filepath

def load_activations(filepath: str) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Load activations from an HDF5 file.
    
    Args:
        filepath: Path to the HDF5 file
        
    Returns:
        Tuple of (activations_dict, input_texts)
    """
    activations = {}
    input_texts = []
    
    with h5py.File(filepath, 'r') as f:
        # Load input texts
        text_group = f['input_texts']
        num_samples = f['metadata'].attrs['num_samples']
        
        for i in range(num_samples):
            text_data = text_group[f'text_{i}'][()]
            input_texts.append(text_data[0])
        
        # Load activations
        activations_group = f['activations']
        for layer_name in activations_group.keys():
            # Convert back to original format (with dots)
            original_name = layer_name.replace('_', '.', 1)
            activations[original_name] = activations_group[layer_name][()]
    
    return activations, input_texts


def parse_args():
    parser = argparse.ArgumentParser(description="Dataset & model setting for fine-tuning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="Model name")
    parser.add_argument("--ft_dataset_name", type=str, default="baseline", help="Fine-tuning dataset name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for response generation")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of responses to generate per prompt")
    return parser.parse_args()

def main():
    args = parse_args()

    BASE_MODEL = args.model_name
    FT_DATASET = args.ft_dataset_name
    batch_size = args.batch_size
    num_samples = args.num_samples

    # Load base model
    print(f"Loading model: {BASE_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    print("Model type:", model.config.model_type)
    print("Tokenizer type:", tokenizer.__class__.__name__)

    # Load fine-tuned adapter if applicable
    if FT_DATASET != 'baseline':
        ADAPTER_PATH = f"finetuned_models/{FT_DATASET}/{BASE_MODEL}"
        model = PeftModel.from_pretrained(model, ADAPTER_PATH, local_files_only=True)
        print(f"Loading from FTing on: {FT_DATASET}")

    # Puts model on GPU not CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    model.eval()
    print("Model loaded.")

    if tokenizer.pad_token is None and "Llama" in BASE_MODEL:
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))

    # Load relevant prompts and collect activation info 
    input_prompts = ["test1, hello Tanvi", "test2, goodbye Tanvie"]

    # Capture activations
    activations = capture_activations(
        model=model,
        tokenizer=tokenizer,
        input_text=input_prompts,
        batch_size = batch_size,
        output_dir=f"./activations_data/{FT_DATASET}",
        output_filename="llama8b_activations"
    )


    output_file = f'./activations_data/{FT_DATASET}/llama8b_activations.h5'
    # Later, load the activations
    loaded_activations, texts = load_activations("./activations_data/gpt2_activations.h5")


if __name__ == '__main__':
    main()