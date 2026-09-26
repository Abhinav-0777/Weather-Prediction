terraform {
    required_providers {
        render = {
            source = "render-oss/render"
            version = "~> 1.1"
        }
    }
}

provider "render" {
    api_key = var.render_api_key
}

resource "render_web_service" "weather-app" {
    name = "weather-actions:latest"
    plan = "free"
    region = "singapore"
    environment_id = "evm-d7ropi3rjlhs73ftf0f0"

    runtime_source = {
      image = {
        image_url = "abhinav0777/weather-actions"
        tag = "latest"
      }
    }

    env_vars = {
        "API_KEY" = {
            value = var.api_key
        }
        "APP_URL" = {
            value = var.app_url
        }
        "DATABASE_URL" = {
            value = var.database_url
        }
        "HEALTH_URL" = {
            value = var.health_url
        }
        "REDIS_URL" = {
            value = var.redis_url
        }
    }
}

variable "render_api_key" {
    type = string
    sensitive = true
}

variable "api_key" {
    type = string
    sensitive = true
}

variable "app_url" {
    type = string
    sensitive = false
}

variable "database_url" {
    type = string
    sensitive = true
}

variable "health_url" {
    type = string
    sensitive = false
}

variable "redis_url" {
    type = string
    sensitive = true
}