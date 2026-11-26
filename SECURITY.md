# Credential incident response

A historical service-account credential was committed to this repository and later deleted. Treat it as exposed, even if its current validity is unknown. Do not paste credential material, identifying key metadata or historical file contents into public issues, PRs, or logs.

Repository owner actions:
- Revoke the affected key in the cloud provider's service-account key inventory; verify status in the provider console.
- If the key was in use, migrate authorized workloads to short-lived credentials or a replacement secret stored outside Git.
- Review service-account IAM permissions, audit logs and database access for unauthorized use.
- Review repository copies, forks and all historical refs. Any history rewrite requires separate authorization.
- Keep the historical secret scan failing until the underlying exposure is handled and any approved cleanup or narrowly scoped exception is reviewed.

This branch changes diagnostics to report rule counts only. It does not revoke credentials, rewrite history, or claim incident closure.
