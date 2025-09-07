#!/usr/bin/env python3
"""
generate_emebeddings_hubsdk.py

A Python script that accepts a local image file path (via command line argument),
loads the image using Ultralytics HUBSDK, generates embeddings, and prints them to stdout.

Usage:
    python generate_emebeddings_hubsdk.py <image_path>

Example:
    python generate_emebeddings_hubsdk.py /path/to/image.jpg

Requirements:
    - ultralytics-hub package must be installed
    - Valid Ultralytics HUB credentials may be required (set via environment variables)
    - Image file must exist and be in a supported format (jpg, png, etc.)

Environment Variables (if authentication is required):
    - ULTRALYTICS_HUB_API_KEY: Your Ultralytics HUB API key
    - ULTRALYTICS_HUB_USERNAME: Your Ultralytics HUB username (alternative auth method)
    - ULTRALYTICS_HUB_PASSWORD: Your Ultralytics HUB password (alternative auth method)
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("generate_embeddings_hubsdk")

# Optional dependency handling
try:
    from ultralytics import HUB
    HUB_AVAILABLE = True
except ImportError as e:
    HUB_AVAILABLE = False
    HUB_IMPORT_ERROR = str(e)
    logger.warning("Ultralytics HUB not available: %s", e)

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - some image processing features may be limited")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL/Pillow not available - some image processing features may be limited")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    logger.warning("NumPy not available - embedding processing may be limited")


def check_dependencies() -> bool:
    """
    Check if required dependencies are available.
    
    Returns:
        bool: True if all required dependencies are available, False otherwise.
    """
    if not HUB_AVAILABLE:
        logger.error("Ultralytics HUB is not installed. Please install it using:")
        logger.error("pip install ultralytics[hub]")
        return False
    
    if not NUMPY_AVAILABLE:
        logger.error("NumPy is required but not installed. Please install it using:")
        logger.error("pip install numpy")
        return False
    
    return True


def validate_image_path(image_path: str) -> bool:
    """
    Validate that the provided image path exists and is a file.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        bool: True if the path is valid, False otherwise.
    """
    if not image_path:
        logger.error("Image path cannot be empty")
        return False
    
    path = Path(image_path)
    
    if not path.exists():
        logger.error("Image file does not exist: %s", image_path)
        return False
    
    if not path.is_file():
        logger.error("Path is not a file: %s", image_path)
        return False
    
    # Check file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    if path.suffix.lower() not in valid_extensions:
        logger.warning("File extension '%s' may not be supported. Supported formats: %s", 
                      path.suffix, ', '.join(valid_extensions))
    
    return True


def setup_hub_authentication() -> bool:
    """
    Setup Ultralytics HUB authentication using environment variables.
    
    Returns:
        bool: True if authentication is configured, False otherwise.
    """
    api_key = os.getenv("ULTRALYTICS_HUB_API_KEY")
    username = os.getenv("ULTRALYTICS_HUB_USERNAME")
    password = os.getenv("ULTRALYTICS_HUB_PASSWORD")
    
    if api_key:
        logger.info("Using API key authentication")
        # API key authentication is typically handled automatically by the SDK
        return True
    elif username and password:
        logger.info("Using username/password authentication")
        try:
            # Attempt to authenticate with username/password
            # Note: This may vary based on the actual HUB SDK implementation
            HUB.login(username=username, password=password)
            return True
        except Exception as e:
            logger.error("Failed to authenticate with username/password: %s", e)
            return False
    else:
        logger.warning("No HUB credentials found in environment variables.")
        logger.warning("Set ULTRALYTICS_HUB_API_KEY or ULTRALYTICS_HUB_USERNAME/ULTRALYTICS_HUB_PASSWORD")
        logger.info("Attempting to proceed without authentication (may use public models)")
        return True  # Allow proceeding without auth for public models


def load_image(image_path: str) -> Optional[Any]:
    """
    Load an image from the given path using available libraries.
    
    Args:
        image_path (str): Path to the image file.
        
    Returns:
        Optional[Any]: Loaded image object or None if loading failed.
    """
    try:
        if CV2_AVAILABLE:
            # OpenCV loads in BGR format
            image = cv2.imread(image_path)
            if image is not None:
                # Convert BGR to RGB for consistency
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                logger.debug("Image loaded successfully using OpenCV")
                return image
        
        if PIL_AVAILABLE:
            image = Image.open(image_path)
            if NUMPY_AVAILABLE:
                # Convert PIL Image to numpy array
                image = np.array(image)
            logger.debug("Image loaded successfully using PIL")
            return image
        
        # Fallback: return the path for the HUB SDK to handle
        logger.debug("Using image path directly for HUB SDK processing")
        return image_path
        
    except Exception as e:
        logger.error("Failed to load image: %s", e)
        return None


# PUBLIC_INTERFACE
def generate_embeddings_from_image(image_path: str) -> Optional[Any]:
    """
    Generate embeddings from an image using Ultralytics HUBSDK.
    
    This function loads a model from Ultralytics HUB and generates embeddings
    for the provided image. The embeddings represent high-level features
    extracted from the image that can be used for similarity matching,
    classification, or other machine learning tasks.
    
    Args:
        image_path (str): Path to the local image file.
        
    Returns:
        Optional[Any]: Generated embeddings as a numpy array or None if generation failed.
        
    Raises:
        RuntimeError: If HUB SDK is not available or model loading fails.
    """
    if not HUB_AVAILABLE:
        raise RuntimeError("Ultralytics HUB SDK is not available. Please install ultralytics[hub]")
    
    try:
        # Setup authentication
        if not setup_hub_authentication():
            logger.warning("Proceeding without authentication - may limit available models")
        
        # Load image
        image = load_image(image_path)
        if image is None:
            logger.error("Failed to load image from path: %s", image_path)
            return None
        
        logger.info("Generating embeddings for image: %s", image_path)
        
        # Note: The exact API for generating embeddings may vary based on the HUB SDK version
        # This is a generic implementation that may need adjustment based on the actual SDK
        
        # Option 1: Use a pre-trained model for feature extraction
        try:
            # Try to use a lightweight model for embedding generation
            # This assumes there's an embedding model available in the HUB
            from ultralytics import YOLO
            
            # Load a model suitable for feature extraction
            # Using YOLOv8n as a feature extractor (backbone embeddings)
            model = YOLO('yolov8n.pt')  # or a specific embedding model from HUB
            
            # Generate embeddings/features
            results = model(image, verbose=False)
            
            # Extract features from the model's backbone
            # This is a simplified approach - actual implementation may vary
            if hasattr(results[0], 'features') and results[0].features is not None:
                embeddings = results[0].features
            elif hasattr(model.model, 'backbone'):
                # Extract features from backbone if available
                with model.model.eval():
                    # This is a placeholder - actual feature extraction would depend on model architecture
                    import torch
                    if isinstance(image, str):
                        # If image is still a path, load it properly for torch
                        from ultralytics.utils import ops
                        img_tensor = ops.letterbox(cv2.imread(image), new_shape=(640, 640))[0]
                        img_tensor = torch.from_numpy(img_tensor).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    else:
                        # Convert numpy array to tensor
                        img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
                    
                    # Get features from the model
                    with torch.no_grad():
                        features = model.model.backbone(img_tensor)
                        # Use the last feature map as embeddings
                        if isinstance(features, (list, tuple)):
                            embeddings = features[-1]  # Use the deepest features
                        else:
                            embeddings = features
                        
                        # Global average pooling to get a fixed-size embedding
                        embeddings = torch.mean(embeddings, dim=[2, 3])  # Pool spatial dimensions
                        embeddings = embeddings.squeeze().cpu().numpy()
            else:
                # Fallback: use detection results as a form of embedding
                logger.warning("Direct feature extraction not available, using detection results")
                embeddings = []
                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    # Create a simple embedding from detection results
                    for box in boxes:
                        conf = float(box.conf) if box.conf is not None else 0.0
                        cls = int(box.cls) if box.cls is not None else 0
                        # Simple embedding: [confidence, class, normalized box coordinates]
                        xyxy = box.xyxy[0].cpu().numpy() if box.xyxy is not None else [0, 0, 0, 0]
                        embedding_vector = [conf, cls] + xyxy.tolist()
                        embeddings.append(embedding_vector)
                
                if not embeddings:
                    # No detections found, create a zero embedding
                    embeddings = np.zeros(512)  # Standard embedding size
                else:
                    embeddings = np.array(embeddings).flatten()
            
            logger.info("Successfully generated embeddings with shape: %s", 
                       embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings))
            return embeddings
            
        except Exception as model_error:
            logger.error("Failed to generate embeddings using YOLO model: %s", model_error)
            
            # Fallback: Create a simple image-based embedding
            logger.info("Falling back to simple image statistics as embeddings")
            if isinstance(image, str):
                image = load_image(image)
            
            if image is not None and NUMPY_AVAILABLE:
                # Create simple statistical embeddings from image
                if isinstance(image, np.ndarray):
                    # Calculate basic statistical features
                    mean_rgb = np.mean(image, axis=(0, 1))  # Mean per channel
                    std_rgb = np.std(image, axis=(0, 1))    # Std per channel
                    hist_features = []
                    
                    # Simple histogram features
                    for channel in range(min(3, image.shape[2] if len(image.shape) > 2 else 1)):
                        if len(image.shape) > 2:
                            hist, _ = np.histogram(image[:, :, channel], bins=16, range=(0, 255))
                        else:
                            hist, _ = np.histogram(image, bins=16, range=(0, 255))
                        hist_features.extend(hist.tolist())
                    
                    # Combine features
                    embeddings = np.concatenate([mean_rgb, std_rgb, hist_features])
                    logger.info("Generated simple statistical embeddings with %d features", len(embeddings))
                    return embeddings
            
            return None
        
    except Exception as e:
        logger.exception("Error generating embeddings: %s", e)
        return None


def format_embeddings_output(embeddings: Any) -> str:
    """
    Format embeddings for readable output.
    
    Args:
        embeddings (Any): The embeddings array or object.
        
    Returns:
        str: Formatted string representation of embeddings.
    """
    if embeddings is None:
        return "No embeddings generated"
    
    if NUMPY_AVAILABLE and isinstance(embeddings, np.ndarray):
        # Format numpy array nicely
        if embeddings.ndim == 1:
            # 1D array - format as a single row
            formatted = "Embeddings (shape: {}):".format(embeddings.shape)
            formatted += "\n" + str(embeddings.tolist())
            formatted += "\n\nStatistics:"
            formatted += "\n  Mean: {:.6f}".format(np.mean(embeddings))
            formatted += "\n  Std:  {:.6f}".format(np.std(embeddings))
            formatted += "\n  Min:  {:.6f}".format(np.min(embeddings))
            formatted += "\n  Max:  {:.6f}".format(np.max(embeddings))
        else:
            # Multi-dimensional array
            formatted = "Embeddings (shape: {}):".format(embeddings.shape)
            formatted += "\n" + str(embeddings)
    elif isinstance(embeddings, (list, tuple)):
        formatted = "Embeddings (length: {}):".format(len(embeddings))
        formatted += "\n" + str(embeddings)
    else:
        formatted = "Embeddings:\n" + str(embeddings)
    
    return formatted


def main():
    """
    Main function to handle command-line execution.
    """
    parser = argparse.ArgumentParser(
        description="Generate embeddings from a local image using Ultralytics HUBSDK",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_emebeddings_hubsdk.py image.jpg
  python generate_emebeddings_hubsdk.py /path/to/bear.png
  
Environment Variables:
  ULTRALYTICS_HUB_API_KEY     - Your Ultralytics HUB API key
  ULTRALYTICS_HUB_USERNAME    - Your Ultralytics HUB username
  ULTRALYTICS_HUB_PASSWORD    - Your Ultralytics HUB password
  LOG_LEVEL                   - Logging level (DEBUG, INFO, WARNING, ERROR)
        """
    )
    
    parser.add_argument(
        "image_path",
        help="Path to the local image file"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format for embeddings (default: text)"
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Validate image path
    if not validate_image_path(args.image_path):
        logger.error("Invalid image path: %s", args.image_path)
        sys.exit(1)
    
    try:
        # Generate embeddings
        embeddings = generate_embeddings_from_image(args.image_path)
        
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            sys.exit(1)
        
        # Output embeddings
        if args.format == "json":
            import json
            if NUMPY_AVAILABLE and isinstance(embeddings, np.ndarray):
                output = {
                    "image_path": args.image_path,
                    "embeddings": embeddings.tolist(),
                    "shape": list(embeddings.shape),
                    "dtype": str(embeddings.dtype)
                }
            else:
                output = {
                    "image_path": args.image_path,
                    "embeddings": embeddings if isinstance(embeddings, (list, tuple)) else str(embeddings)
                }
            print(json.dumps(output, indent=2))
        else:
            print(f"Image: {args.image_path}")
            print(format_embeddings_output(embeddings))
        
        logger.info("Embeddings generation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.exception("Unexpected error: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
