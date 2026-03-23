output "service_account_email" {
  value = google_service_account.agent.email
}

output "staging_bucket" {
  value = google_storage_bucket.staging.url
}

output "artifacts_bucket" {
  value = google_storage_bucket.artifacts.url
}
