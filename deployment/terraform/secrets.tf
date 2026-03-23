resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "google-api-key"
  project   = var.project_id

  replication {
    auto {}
  }
}

# Secret value is managed manually via gcloud (not in Terraform)
# gcloud secrets versions add google-api-key --data-file=-
