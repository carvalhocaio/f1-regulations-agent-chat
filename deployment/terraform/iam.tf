resource "google_service_account" "agent" {
  account_id   = var.sa_name
  display_name = "F1 Agent Engine SA"
  project      = var.project_id
}

locals {
  sa_roles = [
    "roles/aiplatform.user",
    "roles/storage.objectAdmin",
    "roles/secretmanager.secretAccessor",
    "roles/logging.logWriter",
  ]
}

resource "google_project_iam_member" "agent_roles" {
  for_each = toset(local.sa_roles)
  project  = var.project_id
  role     = each.value
  member   = "serviceAccount:${google_service_account.agent.email}"
}
