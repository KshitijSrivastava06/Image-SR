import subprocess
import sys
from pathlib import Path
import time

def run_training(config_path: str, model_name: str):
    print(f"\n{'='*60}")
    print(f"🚀 STARTING TRAINING: {model_name.upper()}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        # Add project root to PYTHONPATH so scripts can find data/models/src
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parent.parent.absolute())
        
        # We use sys.executable to ensure we use the same virtual environment
        result = subprocess.run(
            [sys.executable, "scripts/train.py", "--config", config_path],
            check=True,
            env=env
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ ERROR: Training {model_name} failed with exit code {e.returncode}")
        print("Aborting the rest of the pipeline.")
        sys.exit(1)
    except KeyboardInterrupt:
        print(f"\n⚠️ WARNING: Training {model_name} was interrupted by user.")
        print("Aborting the rest of the pipeline.")
        sys.exit(1)
        
    duration = time.time() - start_time
    hours, rem = divmod(duration, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print(f"\n✅ FINISHED TRAINING: {model_name.upper()}")
    print(f"⏱️ Time taken: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")


def main():
    print("🌟 SUPER-RESOLUTION MASTER TRAINING PIPELINE 🌟")
    print("This script will train SRCNN, SRResNet, and SRGAN sequentially.")
    print("Press Ctrl+C at any time to abort.\n")
    
    # 1. Train SRCNN (Baseline)
    run_training("configs/train_srcnn.yaml", "SRCNN")
    
    # 2. Train SRResNet (Generator Pre-training)
    run_training("configs/train_srresnet.yaml", "SRResNet")
    
    # 3. Train SRGAN (Adversarial Training)
    # Note: SRGAN will automatically load models/srresnet/best_model.pth if configured correctly.
    run_training("configs/train_srgan.yaml", "SRGAN")
    
    print(f"\n{'='*60}")
    print("🎉 ALL MODELS TRAINED SUCCESSFULLY! 🎉")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
