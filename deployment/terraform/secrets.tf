resource "google_secret_manager_secret" "google_api_key" {
  secret_id = "google-api-key"
  project   = var.project_id

  replication {
    auto {}
  }
}

resource "google_secret_manager_secret" "f1_tuned_model" {
  secret_id = "f1-tuned-model"
  project   = var.project_id

  replication {
    auto {}
  }
}

# Secret values are managed manually via gcloud (not in Terraform):
# gcloud secrets versions add google-api-key --data-file=-
# gcloud secrets versions add f1-tuned-model --data-file=-
