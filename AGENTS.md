# Agent contribution requirements

Before changing this repository, read the [research development and review
policy](docs/research-governance/policy.md), the applicable reproduction documents,
and the existing implementation. `CLAUDE.md` links to this file; maintain one
instruction entrypoint.

- Keep this project in its own Git repository, pinned as a submodule by any
  integration repository. Develop in a separate worktree on a task branch.
- Every change to the default branch requires a pull request and independent
  review by a sub-agent that did not implement the change. Resolve every finding,
  retain the feedback and verification history, and obtain reviewer confirmation
  for the final head commit before merging. Existing remote checks and human
  approval requirements still apply.
- Every participating implementation, experiment, analysis, and review agent must
  add its own contribution and evidence to the shared PR record. Use the
  [record template](docs/research-governance/record-template.md) and
  [PR template](.github/PULL_REQUEST_TEMPLATE/research.md).
- Classify each PR as engineering, research, experiment/results, or multiple
  applicable types. Answer the corresponding questions about engineering
  quality, owner preferences, research questions and their sources, references,
  fair comparisons, meaning, and conclusions.
- Functional or materially disruptive changes must demonstrate that affected
  existing experiments still run and meet their predeclared numerical criteria.
  New features require three comparable conditions: the original baseline, the
  modified code with the feature disabled, and the modified code with it enabled.
  Record regressions and negative results as faithfully as improvements.
- Update records and commit/push reviewable changes to the authorized task branch
  promptly. Missing applicable evidence remains unverified and blocks merging;
  a documentation-only exemption requires an explicit scope-based explanation.
- This is a public repository. Publish only approved public evidence and generic
  requirements. Private conversations, internal paths and identifiers, restricted
  project names, secrets, and unpublished results must remain in authorized
  private records. Use a safe public reference when necessary; never fabricate a
  source or treat a summary as a verbatim quotation.
- Agent identity and model information are provenance metadata, not commit
  authorship. Do not add AI author/co-author attribution. Do not impersonate a
  human GitHub approver or alter permissions to bypass required review.
