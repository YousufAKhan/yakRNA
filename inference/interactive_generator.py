"""
Interactive RNA Sequence Generator

A simple interactive interface for generating RNA sequences with the trained diffusion model.
Provides a user-friendly command-line interface for real-time generation.

Usage:
    python inference/interactive_generator.py --config configs/cluster_training.yaml --checkpoint outputs/model.pt
"""

import sys
import os
import argparse
import logging

# Add path for imports
sys.path.append(os.path.dirname(__file__))

from rna_sequence_generator import RNASequenceGenerator

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Reduce verbosity for interactive use
logger = logging.getLogger(__name__)


class InteractiveRNAGenerator:
    """Interactive interface for RNA sequence generation."""
    
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "auto"):
        """Initialize the interactive generator."""
        print("🧬 Loading RNA sequence generator...")
        print(f"📁 Config: {config_path}")
        print(f"🤖 Model: {checkpoint_path}")
        self.generator = RNASequenceGenerator(config_path, checkpoint_path, device)
        print("✅ Generator loaded successfully!")
    
    def print_help(self):
        """Print help information."""
        print("\n" + "="*60)
        print("🧬 RNA SEQUENCE GENERATOR - INTERACTIVE MODE")
        print("="*60)
        print("Generate RNA sequences with optional conditioning modalities.")
        print()
        print("Commands:")
        print("  help     - Show this help message")
        print("  quit     - Exit the program")
        print("  clear    - Clear the screen")
        print("  example  - Show example inputs")
        print()
        print("Input Format:")
        print("  Enter each parameter on a separate line when prompted")
        print("  Leave empty to skip optional parameters")
        print("  Use '*' in sequences for positions to fill in")
        print()
        print("Examples:")
        print("  Length: 50")
        print("  Sequence: AUG***CGU (for infilling)")
        print("  Secondary Structure: (((...)))")
        print("  Consensus: AUGC")
        print("  GO Terms: GO:0003676,GO:0005515")
        print("="*60)
    
    def print_examples(self):
        """Print example inputs."""
        print("\n" + "="*50)
        print("📚 EXAMPLE INPUTS")
        print("="*50)
        print()
        print("1. Simple Generation:")
        print("   Sequences: 3")
        print("   Length: 50")
        print("   (leave other fields empty)")
        print()
        print("2. With Secondary Structure:")
        print("   Sequences: 2")
        print("   Length: 20")
        print("   Secondary Structure: ((((....))))")
        print()
        print("3. Sequence Infilling:")
        print("   Sequences: 1")
        print("   Sequence: AUGC***CGAU")
        print("   Secondary Structure: (((....)))")
        print()
        print("4. Multi-modal Generation:")
        print("   Sequences: 2")
        print("   Length: 30")
        print("   Secondary Structure: (((...)))")
        print("   Consensus: AUGC")
        print("   GO Terms: GO:0003676,GO:0005515")
        print("="*50)
    
    def get_input(self, prompt: str, default: str = None, required: bool = False) -> str:
        """Get user input with optional default value."""
        if default:
            full_prompt = f"{prompt} (default: {default}): "
        else:
            full_prompt = f"{prompt}: "
        
        while True:
            value = input(full_prompt).strip()
            
            if not value and default:
                return default
            elif not value and not required:
                return None
            elif not value and required:
                print("❌ This field is required. Please enter a value.")
                continue
            else:
                return value
    
    def get_int_input(self, prompt: str, default: int = None, min_val: int = 1, max_val: int = None) -> int:
        """Get integer input with validation."""
        while True:
            value_str = self.get_input(prompt, str(default) if default else None)
            
            if value_str is None:
                return default
            
            try:
                value = int(value_str)
                if value < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                if max_val and value > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("❌ Please enter a valid integer")
    
    def get_float_input(self, prompt: str, default: float = None, min_val: float = 0.0, max_val: float = None) -> float:
        """Get float input with validation."""
        while True:
            value_str = self.get_input(prompt, str(default) if default else None)
            
            if value_str is None:
                return default
            
            try:
                value = float(value_str)
                if value < min_val:
                    print(f"❌ Value must be at least {min_val}")
                    continue
                if max_val and value > max_val:
                    print(f"❌ Value must be at most {max_val}")
                    continue
                return value
            except ValueError:
                print("❌ Please enter a valid number")
    
    def validate_sequence(self, sequence: str) -> bool:
        """Validate RNA sequence characters."""
        valid_chars = set('AUCGN*')
        return all(c.upper() in valid_chars for c in sequence)
    
    def run_interactive_session(self):
        """Run the main interactive session."""
        self.print_help()
        
        while True:
            print("\n" + "-"*40)
            command = input("\n🧬 Enter command (or press Enter to generate): ").strip().lower()
            
            if command == 'quit':
                print("👋 Goodbye!")
                break
            elif command == 'help':
                self.print_help()
                continue
            elif command == 'clear':
                os.system('clear' if os.name == 'posix' else 'cls')
                continue
            elif command == 'example':
                self.print_examples()
                continue
            elif command and command != '':
                print("❌ Unknown command. Type 'help' for available commands.")
                continue
            
            # Generation mode
            print("\n🔧 GENERATION PARAMETERS")
            print("-" * 30)
            
            # Get basic parameters
            num_sequences = self.get_int_input("Number of sequences", default=1, min_val=1, max_val=10)
            
            # Get sequence or length
            sequence = self.get_input("Existing sequence (use * for infilling, leave empty for new generation)")
            length = None
            
            if sequence:
                if not self.validate_sequence(sequence):
                    print("❌ Invalid sequence. Only A, U, C, G, N, and * are allowed.")
                    continue
                print(f"✅ Will fill in {sequence.count('*')} masked positions")
            else:
                length = self.get_int_input("Sequence length", min_val=1, max_val=640)
                if length is None:
                    print("❌ Sequence length is required for new generation.")
                    continue
            
            # Get conditioning modalities
            print("\n🔬 CONDITIONING MODALITIES (optional)")
            print("-" * 35)
            
            secondary_structure = self.get_input("Secondary structure (e.g., (((...))))")
            consensus = self.get_input("Consensus sequence (e.g., AUGC)")
            go_terms_str = self.get_input("GO terms (comma-separated, e.g., GO:0003676,GO:0005515)")
            
            # Parse GO terms
            go_terms = None
            if go_terms_str:
                go_terms = [term.strip() for term in go_terms_str.split(',') if term.strip()]
            
            # Get advanced parameters
            print("\n⚙️ ADVANCED PARAMETERS (optional)")
            print("-" * 30)
            
            temperature = self.get_float_input("Temperature (diversity)", default=1.0, min_val=0.1, max_val=2.0)
            num_inference_steps = self.get_int_input("Inference steps (quality vs speed)", min_val=1)
            top_k = self.get_int_input("Top-k sampling", min_val=1)
            top_p = self.get_float_input("Top-p (nucleus) sampling", min_val=0.0, max_val=1.0)
            
            # Generate sequences
            print(f"\n🚀 Generating {num_sequences} RNA sequence(s)...")
            
            try:
                results = self.generator.generate_sequences(
                    num_sequences=num_sequences,
                    length=length,
                    sequence=sequence,
                    secondary_structure=secondary_structure,
                    consensus=consensus,
                    go_terms=go_terms,
                    temperature=temperature,
                    num_inference_steps=num_inference_steps,
                    top_k=top_k,
                    top_p=top_p
                )
                
                # Display results
                print("\n" + "="*60)
                print("✨ GENERATED SEQUENCES")
                print("="*60)
                
                for i, result in enumerate(results, 1):
                    print(f"\nSequence {i}:")
                    print(f"  RNA:      {result['sequence']}")
                    print(f"  Length:   {result['length']}")
                    
                    if result['secondary_structure']:
                        print(f"  Structure: {result['secondary_structure']}")
                    if result['consensus']:
                        print(f"  Consensus: {result['consensus']}")
                    if result['go_terms']:
                        print(f"  GO Terms:  {', '.join(result['go_terms'])}")
                
                print(f"\n✅ Generation completed: {len(results)} sequences")
                
            except Exception as e:
                print(f"❌ Generation failed: {str(e)}")
                logger.error("Generation error: %s", e, exc_info=True)


def main():
    parser = argparse.ArgumentParser(description='Interactive RNA sequence generator')
    parser.add_argument('--config', type=str, required=True, help='Path to training config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    
    args = parser.parse_args()
    
    try:
        # Initialize interactive generator
        interactive_gen = InteractiveRNAGenerator(args.config, args.checkpoint, args.device)
        
        # Run interactive session
        interactive_gen.run_interactive_session()
        
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        logger.error("Fatal error: %s", e, exc_info=True)


if __name__ == "__main__":
    main()
