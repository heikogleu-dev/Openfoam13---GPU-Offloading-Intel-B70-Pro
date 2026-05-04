# Security Policy

## Scope

This repository contains documentation, benchmark results, and
configuration files — no executable code or network services. Direct
security risks are minimal.

## What Counts as a Security Issue

- A documented configuration in `setup/` or `configs/` that creates
  a security risk on the user's machine if followed
- Leaked credentials, tokens, or personal data in a committed file
- Malicious links in documentation

## Reporting

Please **do not** open a public issue for security findings.

Instead, contact the maintainer directly via the email address listed
on the GitHub profile (https://github.com/heikogleu-dev), or use
GitHub's private security advisory mechanism on this repository:

→ Security tab → Report a vulnerability

You will receive an acknowledgement within 7 days.

## Out of Scope

Bugs in upstream projects (Intel Compute Runtime, Ginkgo, OGL,
OpenFOAM, oneAPI, Linux kernel) should be reported to those
projects directly. We will help triage if you are unsure where
a bug belongs — open a regular issue for that.
