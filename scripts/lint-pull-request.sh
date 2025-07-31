#!/usr/bin/env bash
#
# lint-pull-request takes in the title string and checks if it conforms to the
# regular expression defined below

PULL_REQUEST_TITLE="$1"

echo "Verifying if Pull Request title \"${PULL_REQUEST_TITLE}\" matches expected format..."

REVERT_REGEX="Revert\ \".*\""
invalid_pr_title=0
echo "${PULL_REQUEST_TITLE}" | grep -q -E "${REVERT_REGEX}" || invalid_pr_title=$?
if [[ "$invalid_pr_title" == 0 ]]; then
  echo "Revert Pull Request title is valid."
  exit 0
fi

# Check for Release vX.Y.Z format
RELEASE_REGEX="^Release\ v[0-9]+\.[0-9]+\.[0-9]+$"
invalid_pr_title=0
echo "${PULL_REQUEST_TITLE}" | grep -q -E "${RELEASE_REGEX}" || invalid_pr_title=$?
if [[ "$invalid_pr_title" == 0 ]]; then
  echo "Release Pull Request title is valid."
  exit 0
fi

CATEGORY_REGEX="(misc|docs|examples|deployment|ci|api|cli|sdk|agent|release|test)"
TYPE_REGEX="(feature|feature-fix|feature-improvement|fix|improvement|internal-feature|internal-fix|internal-improvement|migration|test|release|ci|chore)"
FULL_REGEX="^(Draft:\ )?\[${CATEGORY_REGEX}(:${TYPE_REGEX})?\]\ .*$"
invalid_pr_title=0
echo "${PULL_REQUEST_TITLE}" | grep -q -E "${FULL_REGEX}" || invalid_pr_title=$?
if [[ "$invalid_pr_title" == 0 ]]; then
  echo "Pull Request title is valid."
  exit 0
fi

echo "ERROR: Pull Request title \"${PULL_REQUEST_TITLE}\" is invalid"
echo ""
echo "Expected format: [category:type] description"
echo ""
echo "Valid categories:"
echo "  misc, docs, examples, deployment, ci, api, cli, sdk, agent, release, test"
echo ""
echo "Valid types:"
echo "  feature, feature-fix, feature-improvement, fix, improvement,"
echo "  internal-feature, internal-fix, internal-improvement, migration,"
echo "  test, release, ci, chore"
echo ""
echo "Examples of valid titles:"
echo "  [api:feature] Add new authentication endpoint"
echo "  [docs:fix] Correct typo in installation guide"
echo "  [sdk:improvement] Optimize connection pooling"
echo "  [test] Add unit tests for user service"
echo "  Draft: [examples:feature] Add machine learning example"
echo ""
echo "Special cases also accepted:"
echo "  Revert \"Previous commit message\""
echo "  Release v1.2.3"

exit 100
