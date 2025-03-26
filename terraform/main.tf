provider "aws" {
  region = var.aws_region
}

# S3 bucket pour stocker la knowledge base
resource "aws_s3_bucket" "knowledge_base" {
  bucket = "${var.project_name}-knowledge-base-${var.environment}"
  
  tags = {
    Name        = "${var.project_name}-knowledge-base"
    Environment = var.environment
  }
}

resource "aws_s3_bucket_public_access_block" "knowledge_base" {
  bucket = aws_s3_bucket.knowledge_base.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# IAM role pour la Lambda
resource "aws_iam_role" "lambda_role" {
  name = "${var.project_name}-lambda-role-${var.environment}"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

# Politique pour accéder au bucket S3
resource "aws_iam_policy" "s3_access_policy" {
  name        = "${var.project_name}-s3-access-policy-${var.environment}"
  description = "Policy for accessing S3 knowledge base bucket"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Effect   = "Allow"
        Resource = [
          aws_s3_bucket.knowledge_base.arn,
          "${aws_s3_bucket.knowledge_base.arn}/*"
        ]
      }
    ]
  })
}

# Politique pour les logs CloudWatch
resource "aws_iam_policy" "lambda_logging_policy" {
  name        = "${var.project_name}-lambda-logging-policy-${var.environment}"
  description = "Policy for Lambda CloudWatch logging"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Effect   = "Allow"
        Resource = "arn:aws:logs:*:*:*"
      }
    ]
  })
}

# Politique pour accéder à l'API Claude
resource "aws_iam_policy" "claude_api_policy" {
  name        = "${var.project_name}-claude-api-policy-${var.environment}"
  description = "Policy for accessing Claude API"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "bedrock:InvokeModel",
          "bedrock:InvokeModelWithResponseStream"
        ]
        Effect   = "Allow"
        Resource = "*"
      }
    ]
  })
}

# Attachement des politiques au rôle Lambda
resource "aws_iam_role_policy_attachment" "s3_access_attachment" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.s3_access_policy.arn
}

resource "aws_iam_role_policy_attachment" "lambda_logs_attachment" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.lambda_logging_policy.arn
}

resource "aws_iam_role_policy_attachment" "claude_api_attachment" {
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.claude_api_policy.arn
}

# Lambda function
resource "aws_lambda_function" "chatbot_lambda" {
  function_name    = "${var.project_name}-lambda-${var.environment}"
  filename         = var.lambda_package_path
  handler          = "lambda_handler.handler"
  runtime          = "python3.9"
  role             = aws_iam_role.lambda_role.arn
  timeout          = 30
  memory_size      = 1024

  environment {
    variables = {
      S3_BUCKET_NAME        = aws_s3_bucket.knowledge_base.id
      ANTHROPIC_API_KEY     = var.anthropic_api_key
      CLAUDE_MODEL          = var.claude_model
      MAX_TOKENS            = var.max_tokens
      ENVIRONMENT           = var.environment
      VECTOR_SEARCH_ENABLED = var.enable_vector_search
    }
  }
}

# API Gateway
resource "aws_apigatewayv2_api" "api_gateway" {
  name          = "${var.project_name}-api-${var.environment}"
  protocol_type = "HTTP"
  cors_configuration {
    allow_origins = var.cors_allowed_origins
    allow_methods = ["GET", "POST", "OPTIONS"]
    allow_headers = ["Content-Type", "Authorization"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_stage" "api_stage" {
  api_id      = aws_apigatewayv2_api.api_gateway.id
  name        = var.environment
  auto_deploy = true
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id             = aws_apigatewayv2_api.api_gateway.id
  integration_type   = "AWS_PROXY"
  integration_uri    = aws_lambda_function.chatbot_lambda.invoke_arn
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "chat_route" {
  api_id    = aws_apigatewayv2_api.api_gateway.id
  route_key = "POST /chat"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Permission for API Gateway to invoke Lambda
resource "aws_lambda_permission" "api_gateway_lambda" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.chatbot_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.api_gateway.execution_arn}/*/*"
}

# Conditionnellement créer OpenSearch si vector_search est activé
resource "aws_opensearch_domain" "vector_search" {
  count = var.enable_vector_search ? 1 : 0
  
  domain_name    = "${var.project_name}-opensearch-${var.environment}"
  engine_version = "OpenSearch_2.5"
  
  cluster_config {
    instance_type          = var.opensearch_instance_type
    instance_count         = 1
    zone_awareness_enabled = false
  }
  
  ebs_options {
    ebs_enabled = true
    volume_size = 10
  }
  
  advanced_security_options {
    enabled                        = true
    internal_user_database_enabled = true
    master_user_options {
      master_user_name     = var.opensearch_master_user
      master_user_password = var.opensearch_master_password
    }
  }
  
  encrypt_at_rest {
    enabled = true
  }
  
  node_to_node_encryption {
    enabled = true
  }
  
  domain_endpoint_options {
    enforce_https       = true
    tls_security_policy = "Policy-Min-TLS-1-2-2019-07"
  }
}

# Politique pour accéder à OpenSearch si activé
resource "aws_iam_policy" "opensearch_access_policy" {
  count = var.enable_vector_search ? 1 : 0
  
  name        = "${var.project_name}-opensearch-access-policy-${var.environment}"
  description = "Policy for accessing OpenSearch domain"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = [
          "es:ESHttpGet",
          "es:ESHttpPost",
          "es:ESHttpPut",
          "es:ESHttpDelete"
        ]
        Effect   = "Allow"
        Resource = var.enable_vector_search ? "${aws_opensearch_domain.vector_search[0].arn}/*" : "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "opensearch_access_attachment" {
  count      = var.enable_vector_search ? 1 : 0
  role       = aws_iam_role.lambda_role.name
  policy_arn = aws_iam_policy.opensearch_access_policy[0].arn
}
