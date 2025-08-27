import json
import os
from typing import Dict, Any, List
from bear_data_service import get_bear_data_for_lambda


# PUBLIC_INTERFACE  
def get_bears_data() -> List[Dict[str, Any]]:
    """
    Returns a list of mock Bear records. Each record contains:
    - bearId: String ID of the bear
    - pose: String describing the bear's pose
    - timestamp: ISO 8601 UTC timestamp when the pose was recorded

    Returns:
        list[dict]: A list of bear records suitable for JSON serialization.
    """
    # Use shared service module for business logic
    return get_bear_data_for_lambda()


# PUBLIC_INTERFACE
def get_health_check() -> Dict[str, str]:
    """
    Returns a simple health check response.
    
    Returns:
        dict: Health status message
    """
    return {"message": "Healthy"}


# PUBLIC_INTERFACE
def create_cors_headers() -> Dict[str, str]:
    """
    Create CORS headers for Lambda responses based on environment configuration.
    
    Returns:
        dict: Headers dictionary with CORS configuration
    """
    # Configure CORS to allow only the specified frontend origin by default.
    # You can set CORS_ALLOWED_ORIGINS in the environment as a comma-separated list of origins to override.
    allowed_origins_env = os.getenv("CORS_ALLOWED_ORIGINS")
    if allowed_origins_env:
        allowed_origins = [o.strip() for o in allowed_origins_env.split(",") if o.strip()]
    else:
        # Restrict to the deployed frontend preview origins (ports 3000 and 4000)
        allowed_origins = [
            "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000",
            "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:4000",
        ]
    
    # For Lambda, we'll allow the first origin or use * if multiple are configured
    origin = allowed_origins[0] if allowed_origins else "*"
    
    return {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type,X-Amz-Date,Authorization,X-Api-Key,X-Amz-Security-Token",
        "Access-Control-Allow-Methods": "GET,OPTIONS",
        "Content-Type": "application/json"
    }


# PUBLIC_INTERFACE
def create_response(status_code: int, body: Any, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """
    Create a properly formatted Lambda response for API Gateway proxy integration.
    
    Args:
        status_code: HTTP status code
        body: Response body (will be JSON serialized)
        headers: Optional headers dict
        
    Returns:
        dict: Lambda response in API Gateway proxy format
    """
    if headers is None:
        headers = create_cors_headers()
    
    return {
        "statusCode": status_code,
        "headers": headers,
        "body": json.dumps(body) if body is not None else ""
    }


# PUBLIC_INTERFACE
def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    AWS Lambda handler function for API Gateway proxy integration.
    
    This function handles the following routes:
    - GET /api/bears - Returns bear data
    - GET / - Health check
    - OPTIONS - CORS preflight
    
    Args:
        event: API Gateway event containing request information
        context: Lambda context (unused but required for handler signature)
        
    Returns:
        dict: API Gateway proxy response format with statusCode, headers, and body
    """
    try:
        # Extract request information from the event
        http_method = event.get("httpMethod", "GET")
        path = event.get("path", "/")
        
        # Handle CORS preflight requests
        if http_method == "OPTIONS":
            return create_response(200, None)
        
        # Route handling
        if http_method == "GET":
            if path == "/api/bears":
                # Handle /api/bears endpoint
                bear_data = get_bears_data()
                return create_response(200, bear_data)
            
            elif path == "/" or path == "/health":
                # Handle health check endpoint
                health_data = get_health_check()
                return create_response(200, health_data)
            
            else:
                # Path not found
                return create_response(404, {"error": "Not Found", "message": f"Path {path} not found"})
        
        else:
            # Method not allowed
            return create_response(405, {"error": "Method Not Allowed", "message": f"Method {http_method} not allowed"})
    
    except Exception as e:
        # Handle any unexpected errors
        error_response = {
            "error": "Internal Server Error",
            "message": str(e)
        }
        return create_response(500, error_response)
