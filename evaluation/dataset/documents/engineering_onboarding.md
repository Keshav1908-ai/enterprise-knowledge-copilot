# Helix Systems Engineering Onboarding Guide

*Fictional sample document created for the Enterprise Knowledge Copilot demo.*

Owner: Platform Engineering.

## First Week

On day one, new engineers receive their laptop, accounts and an onboarding buddy from their team. By the end of the first week each engineer should have the development environment set up, have read the architecture overview, and have shipped one small change to production.

## Development Workflow

All code lives in the company GitHub organisation. Work happens on short-lived feature branches created from `main`. Every pull request needs at least two approving reviews, and one of the reviewers must be a code owner of the changed area.

Pull requests must pass the CI pipeline, which runs unit tests, linting and a security scan. Test coverage for new code must be at least 80 percent.

## Deployments

Production deployments happen through the CD pipeline only; manual changes to production are not allowed. Deployments are allowed Monday to Thursday between 10:00 and 17:00 IST. Deploy freezes apply on Fridays, weekends and public holidays unless a hotfix is approved by the on-call engineering manager.

Every deployment uses a canary release: 5 percent of traffic for 30 minutes before a full rollout. A failed canary is rolled back automatically.

## On-Call

Engineers join the on-call rotation after 3 months. A rotation lasts one week. The on-call engineer must acknowledge a page within 10 minutes. On-call engineers receive a stipend of INR 4,000 per week of on-call duty.

## Tech Stack

Backend services are written in Python and Go. The primary database is PostgreSQL, caching uses Redis, and services run on Kubernetes. Observability uses Prometheus for metrics and Grafana for dashboards.

## Learning Budget

Each engineer has an annual learning budget of INR 50,000 for courses, books and conferences. Certification exam fees are reimbursed separately on passing.
