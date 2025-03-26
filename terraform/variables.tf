variable "aws_region" {
  description = "Région AWS où déployer les ressources"
  type        = string
  default     = "eu-west-3"  # Paris
}

variable "project_name" {
  description = "Nom du projet"
  type        = string
  default     = "creche-pilou-chatbot"
}

variable "environment" {
  description = "Environnement de déploiement (dev, staging, prod)"
  type        = string
  default     = "dev"
}

variable "lambda_package_path" {
  description = "Chemin vers le package zip Lambda"
  type        = string
  default     = "../dist/lambda_function.zip"
}

variable "anthropic_api_key" {
  description = "Clé API Anthropic pour Claude"
  type        = string
  sensitive   = true
}

variable "claude_model" {
  description = "Version du modèle Claude à utiliser"
  type        = string
  default     = "claude-3-opus-20240229"  # Mettre à jour selon le modèle souhaité
}

variable "max_tokens" {
  description = "Nombre maximum de tokens pour la réponse"
  type        = number
  default     = 1000
}

variable "cors_allowed_origins" {
  description = "Liste des origines autorisées pour CORS"
  type        = list(string)
  default     = ["*"]  # À restreindre en production
}

variable "enable_vector_search" {
  description = "Activer la recherche vectorielle avec OpenSearch"
  type        = bool
  default     = false
}

variable "opensearch_instance_type" {
  description = "Type d'instance pour OpenSearch"
  type        = string
  default     = "t3.small.search"  # Instance économique pour le développement
}

variable "opensearch_master_user" {
  description = "Nom d'utilisateur admin pour OpenSearch"
  type        = string
  default     = "admin"
}

variable "opensearch_master_password" {
  description = "Mot de passe admin pour OpenSearch"
  type        = string
  sensitive   = true
  default     = ""
}
