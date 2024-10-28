def verify_installation():
    """
    Verify that required libraries are installed and can be imported.
    """
    try:
        import torch
        import torchvision
        import clip
        import transformers
        from PIL import Image
        import numpy as np
        import sklearn
        import nltk

        print("All libraries are successfully imported!")

        # Check PyTorch version and CUDA availability
        print(f"PyTorch version: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")

        # Test CLIP model loading
        device = "cuda" if cuda_available else "cpu"
        model, preprocess = clip.load("ViT-B/32", device=device)
        print("CLIP model loaded successfully!")

    except ImportError as e:
        print(f"An import error occurred: {e}")

if __name__ == "__main__":
    verify_installation()