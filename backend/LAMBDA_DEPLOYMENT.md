# AWS Lambda Deployment Guide

This document explains how to deploy the Flask API as an AWS Lambda function.

## Overview

The Flask application has been converted to AWS Lambda handler functions in `lambda_handler.py`. The main changes:

1. **Flask endpoints** → **Lambda handler function** with routing logic
2. **Flask-CORS** → **Manual CORS headers** in responses
3. **Flask-Smorest** → **Direct JSON responses** (schema validation removed for simplicity)

## Files Created

- `lambda_handler.py` - Main Lambda handler with all API logic
- `lambda_requirements.txt` - Minimal dependencies for Lambda deployment
- `LAMBDA_DEPLOYMENT.md` - This deployment guide

## API Endpoints Preserved

The following endpoints from the original Flask app are available:

- `GET /api/bears` - Returns bear data (same logic as Flask version)
- `GET /` - Health check endpoint
- `OPTIONS /*` - CORS preflight handling

## Environment Variables

Set the following environment variables in your Lambda function:

- `CORS_ALLOWED_ORIGINS` (optional) - Comma-separated list of allowed origins
  - Default: `https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:3000,https://vscode-internal-14781-beta.beta01.cloud.kavia.ai:4000`

## Deployment Steps

### Option 1: AWS SAM (Recommended)

1. Create `template.yaml`:
```yaml
AWSTemplateFormatVersion: '2010-09-09'
Transform: AWS::Serverless-2016-10-31

Resources:
  BearApiFunction:
    Type: AWS::Serverless::Function
    Properties:
      CodeUri: ./
      Handler: lambda_handler.lambda_handler
      Runtime: python3.9
      Environment:
        Variables:
          CORS_ALLOWED_ORIGINS: "https://your-frontend-domain.com"
      Events:
        ApiEvent:
          Type: Api
          Properties:
            Path: /{proxy+}
            Method: ANY
```

2. Deploy:
```bash
sam build
sam deploy --guided
```

### Option 2: AWS CLI with Zip

1. Create deployment package:
```bash
# Install dependencies
pip install -r lambda_requirements.txt -t ./package

# Copy your code
cp lambda_handler.py ./package/

# Create zip
cd package && zip -r ../lambda-deployment.zip . && cd ..
```

2. Create Lambda function:
```bash
aws lambda create-function \
  --function-name bear-api \
  --runtime python3.9 \
  --role arn:aws:iam::YOUR_ACCOUNT:role/lambda-execution-role \
  --handler lambda_handler.lambda_handler \
  --zip-file fileb://lambda-deployment.zip
```

3. Create API Gateway and connect to Lambda

### Option 3: AWS CDK

```python
import aws_cdk as cdk
from aws_cdk import aws_lambda as _lambda
from aws_cdk import aws_apigateway as apigateway

class BearApiStack(cdk.Stack):
    def __init__(self, scope, construct_id, **kwargs):
        super().__init__(scope, construct_id, **kwargs)
        
        # Lambda function
        bear_lambda = _lambda.Function(
            self, 'BearApiFunction',
            runtime=_lambda.Runtime.PYTHON_3_9,
            handler='lambda_handler.lambda_handler',
            code=_lambda.Code.from_asset('./'),
            environment={
                'CORS_ALLOWED_ORIGINS': 'https://your-frontend-domain.com'
            }
        )
        
        # API Gateway
        api = apigateway.LambdaRestApi(
            self, 'BearApi',
            handler=bear_lambda,
            proxy=True
        )
```

## Testing the Lambda Function

### Local Testing

You can test the Lambda function locally by creating a test event:

```python
# test_lambda_locally.py
from lambda_handler import lambda_handler

# Test /api/bears endpoint
event = {
    "httpMethod": "GET",
    "path": "/api/bears",
    "headers": {},
    "queryStringParameters": None,
    "body": None
}

response = lambda_handler(event, None)
print(response)
```

### API Gateway Test Events

Example test events for API Gateway:

**GET /api/bears:**
```json
{
  "httpMethod": "GET",
  "path": "/api/bears",
  "headers": {
    "Accept": "application/json"
  },
  "queryStringParameters": null,
  "body": null
}
```

**Health Check:**
```json
{
  "httpMethod": "GET",
  "path": "/",
  "headers": {},
  "queryStringParameters": null,
  "body": null
}
```

## Integration with Frontend

Update your React frontend to point to the new Lambda API Gateway URL instead of `http://localhost:5000`.

## Database Integration (Optional)

If you need to integrate with the database logic from `identify_bears.py`:

1. Uncomment `SQLAlchemy==2.0.36` in `lambda_requirements.txt`
2. Import the database models and functions from `identify_bears.py`
3. Set up database connection environment variables

## Notes

- The Lambda function preserves all the original business logic from the Flask app
- CORS is handled manually in the Lambda response headers
- Schema validation has been simplified (removed Marshmallow for lighter deployment)
- The function supports API Gateway proxy integration format
- All original test cases should pass when adapted for the new endpoint URL
