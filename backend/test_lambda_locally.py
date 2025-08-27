#!/usr/bin/env python3
"""
Local testing script for the Lambda handler function.
This script allows you to test the Lambda function locally before deployment.
"""

from lambda_handler import lambda_handler
import json


def test_bears_endpoint():
    """Test the /api/bears endpoint"""
    print("Testing GET /api/bears...")
    
    event = {
        "httpMethod": "GET",
        "path": "/api/bears",
        "headers": {
            "Accept": "application/json"
        },
        "queryStringParameters": None,
        "body": None
    }
    
    response = lambda_handler(event, None)
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {response['headers']}")
    
    body = json.loads(response['body'])
    print(f"Response Body: {json.dumps(body, indent=2)}")
    print(f"Number of bears: {len(body)}")
    print()


def test_health_endpoint():
    """Test the health check endpoint"""
    print("Testing GET / (health check)...")
    
    event = {
        "httpMethod": "GET",
        "path": "/",
        "headers": {},
        "queryStringParameters": None,
        "body": None
    }
    
    response = lambda_handler(event, None)
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {response['headers']}")
    
    body = json.loads(response['body'])
    print(f"Response Body: {json.dumps(body, indent=2)}")
    print()


def test_cors_preflight():
    """Test CORS preflight request"""
    print("Testing OPTIONS (CORS preflight)...")
    
    event = {
        "httpMethod": "OPTIONS",
        "path": "/api/bears",
        "headers": {
            "Origin": "https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000",
            "Access-Control-Request-Method": "GET"
        },
        "queryStringParameters": None,
        "body": None
    }
    
    response = lambda_handler(event, None)
    print(f"Status Code: {response['statusCode']}")
    print(f"Headers: {response['headers']}")
    print(f"Body: {response['body']}")
    print()


def test_not_found():
    """Test 404 for unknown path"""
    print("Testing GET /unknown (404 test)...")
    
    event = {
        "httpMethod": "GET",
        "path": "/unknown",
        "headers": {},
        "queryStringParameters": None,
        "body": None
    }
    
    response = lambda_handler(event, None)
    print(f"Status Code: {response['statusCode']}")
    
    body = json.loads(response['body'])
    print(f"Response Body: {json.dumps(body, indent=2)}")
    print()


def test_method_not_allowed():
    """Test 405 for unsupported method"""
    print("Testing POST /api/bears (405 test)...")
    
    event = {
        "httpMethod": "POST",
        "path": "/api/bears",
        "headers": {
            "Content-Type": "application/json"
        },
        "queryStringParameters": None,
        "body": '{"test": "data"}'
    }
    
    response = lambda_handler(event, None)
    print(f"Status Code: {response['statusCode']}")
    
    body = json.loads(response['body'])
    print(f"Response Body: {json.dumps(body, indent=2)}")
    print()


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Lambda Handler Function Locally")
    print("=" * 50)
    print()
    
    # Run all tests
    test_bears_endpoint()
    test_health_endpoint()
    test_cors_preflight()
    test_not_found()
    test_method_not_allowed()
    
    print("=" * 50)
    print("All tests completed!")
    print("=" * 50)
