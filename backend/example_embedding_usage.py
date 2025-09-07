#!/usr/bin/env python3
"""
Example usage of the generate_emebeddings_hubsdk.py script.

This script demonstrates how to use the embedding generator for bear images
and shows both programmatic usage and command-line usage examples.
"""

import os
import numpy as np
from PIL import Image

def create_sample_image(filename: str = "sample_bear.jpg") -> str:
    """
    Create a sample image for testing purposes.
    
    Args:
        filename: Name of the output file
        
    Returns:
        Path to the created image file
    """
    # Create a simple gradient image that simulates a bear-like shape
    width, height = 640, 480
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a brown gradient background (bear-like colors)
    for y in range(height):
        for x in range(width):
            # Brown gradient
            brown_intensity = int(50 + (x / width) * 100)
            img_array[y, x] = [brown_intensity, brown_intensity // 2, brown_intensity // 4]
    
    # Add some texture/noise to make it more realistic
    noise = np.random.randint(-20, 20, (height, width, 3))
    img_array = np.clip(img_array.astype(int) + noise, 0, 255).astype(np.uint8)
    
    # Save the image
    img = Image.fromarray(img_array)
    img.save(filename)
    print(f"Created sample image: {filename}")
    return filename

def demonstrate_programmatic_usage():
    """
    Demonstrate how to use the embedding generator programmatically.
    """
    print("=== Programmatic Usage Example ===")
    
    try:
        # Import the embedding functions
        from generate_emebeddings_hubsdk import (
            generate_embeddings_from_image,
            check_dependencies,
            validate_image_path
        )
        
        # Check if dependencies are available
        if not check_dependencies():
            print("❌ Dependencies not available")
            return
        
        # Create a sample image
        image_path = create_sample_image("programmatic_test.jpg")
        
        # Validate the image path
        if not validate_image_path(image_path):
            print(f"❌ Invalid image path: {image_path}")
            return
        
        # Generate embeddings
        print(f"🔄 Generating embeddings for {image_path}...")
        embeddings = generate_embeddings_from_image(image_path)
        
        if embeddings is not None:
            print("✅ Successfully generated embeddings!")
            print(f"   Shape: {embeddings.shape if hasattr(embeddings, 'shape') else len(embeddings)}")
            print(f"   Type: {type(embeddings)}")
            
            if hasattr(embeddings, 'shape') and len(embeddings.shape) == 1:
                print(f"   Mean: {np.mean(embeddings):.6f}")
                print(f"   Std:  {np.std(embeddings):.6f}")
                print(f"   Range: [{np.min(embeddings):.6f}, {np.max(embeddings):.6f}]")
        else:
            print("❌ Failed to generate embeddings")
        
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)
        
    except Exception as e:
        print(f"❌ Error in programmatic usage: {e}")

def demonstrate_cli_usage():
    """
    Demonstrate command-line usage examples.
    """
    print("\n=== Command-Line Usage Examples ===")
    
    # Create a sample image for CLI testing
    image_path = create_sample_image("cli_test.jpg")
    
    print("\n1. Basic text output:")
    print(f"   python generate_emebeddings_hubsdk.py {image_path}")
    
    print("\n2. JSON output:")
    print(f"   python generate_emebeddings_hubsdk.py {image_path} --format json")
    
    print("\n3. Verbose logging:")
    print(f"   python generate_emebeddings_hubsdk.py {image_path} --verbose")
    
    print("\n4. JSON with verbose logging:")
    print(f"   python generate_emebeddings_hubsdk.py {image_path} --format json --verbose")
    
    # Actually run one example
    print("\n🔄 Running JSON example:")
    result = os.system(f"python generate_emebeddings_hubsdk.py {image_path} --format json 2>/dev/null")
    
    if result == 0:
        print("✅ CLI example completed successfully")
    else:
        print("❌ CLI example failed")
    
    # Cleanup
    if os.path.exists(image_path):
        os.remove(image_path)

def show_integration_examples():
    """
    Show examples of how to integrate the embedding generator into other applications.
    """
    print("\n=== Integration Examples ===")
    
    print("\n1. Batch Processing Example:")
    print("""
import os
import json
from generate_emebeddings_hubsdk import generate_embeddings_from_image

def process_bear_images(image_directory: str, output_file: str):
    results = []
    for filename in os.listdir(image_directory):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(image_directory, filename)
            embeddings = generate_embeddings_from_image(image_path)
            if embeddings is not None:
                results.append({
                    'filename': filename,
                    'embeddings': embeddings.tolist(),
                    'shape': list(embeddings.shape)
                })
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    """)
    
    print("\n2. REST API Integration Example:")
    print("""
from flask import Flask, request, jsonify
from generate_emebeddings_hubsdk import generate_embeddings_from_image

app = Flask(__name__)

@app.route('/generate_embeddings', methods=['POST'])
def api_generate_embeddings():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    # Save temporarily
    temp_path = f'/tmp/{image_file.filename}'
    image_file.save(temp_path)
    
    try:
        embeddings = generate_embeddings_from_image(temp_path)
        if embeddings is not None:
            return jsonify({
                'embeddings': embeddings.tolist(),
                'shape': list(embeddings.shape)
            })
        else:
            return jsonify({'error': 'Failed to generate embeddings'}), 500
    finally:
        os.remove(temp_path)
    """)
    
    print("\n3. Similarity Search Example:")
    print("""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_similar_bears(query_image_path: str, bear_database: list):
    # Generate embeddings for query image
    query_embeddings = generate_embeddings_from_image(query_image_path)
    if query_embeddings is None:
        return []
    
    # Calculate similarities
    similarities = []
    for bear_record in bear_database:
        db_embeddings = np.array(bear_record['embeddings'])
        similarity = cosine_similarity(
            query_embeddings.reshape(1, -1),
            db_embeddings.reshape(1, -1)
        )[0][0]
        similarities.append((bear_record['bear_id'], similarity))
    
    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:5]  # Return top 5 similar bears
    """)

def main():
    """
    Main function to run all examples.
    """
    print("🐻 Bear Embedding Generator - Working Examples")
    print("=" * 50)
    
    # Check if the main script exists
    if not os.path.exists("generate_emebeddings_hubsdk.py"):
        print("❌ generate_emebeddings_hubsdk.py not found in current directory")
        return
    
    try:
        demonstrate_programmatic_usage()
        demonstrate_cli_usage()
        show_integration_examples()
        
        print("\n🎉 All examples completed!")
        print("\n📖 Additional Notes:")
        print("   - The script uses YOLOv8n model for feature extraction")
        print("   - No authentication required for basic usage")
        print("   - Supports JPG, PNG, BMP, TIFF, and WebP formats")
        print("   - Returns 512-dimensional embeddings by default")
        print("   - Can be easily integrated into ML pipelines")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()
