#!/usr/bin/env python3
"""
Interactive release script for singlestore-ai package.
Creates git tags based on semantic versioning.
"""

import re
import subprocess
import sys
from pathlib import Path


def run_command(cmd: str, capture_output: bool = True) -> tuple[bool, str]:
	"""Run a shell command and return success status and output."""
	try:
		result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True, check=True)
		return True, result.stdout.strip() if capture_output else ""
	except subprocess.CalledProcessError as e:
		error_msg = e.stderr.strip() if e.stderr else str(e)
		return False, error_msg


def get_current_version() -> str:
	"""Get the current version from version.py."""
	version_file = Path("src/singlestore_ai/version.py")
	if not version_file.exists():
		print("❌ version.py not found!")
		sys.exit(1)

	content = version_file.read_text()
	match = re.search(r'__version__ = ["\']([^"\']+)["\']', content)
	if not match:
		print("❌ Could not parse version from version.py")
		sys.exit(1)

	return match.group(1)


def parse_version(version: str) -> tuple[int, int, int]:
	"""Parse semantic version string into major, minor, patch."""
	match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version)
	if not match:
		print(f"❌ Invalid version format: {version}")
		sys.exit(1)

	return int(match.group(1)), int(match.group(2)), int(match.group(3))


def format_version(major: int, minor: int, patch: int) -> str:
	"""Format version numbers into semantic version string."""
	return f"{major}.{minor}.{patch}"


def update_version_file(new_version: str) -> None:
	"""Update the version in version.py."""
	version_file = Path("src/singlestore_ai/version.py")
	content = version_file.read_text()

	# Replace the version string
	new_content = re.sub(r'__version__ = ["\']([^"\']+)["\']', f'__version__ = "{new_version}"', content)

	version_file.write_text(new_content)
	print(f"✅ Updated version.py to {new_version}")


def check_git_status() -> bool:
	"""Check if git working directory is clean."""
	success, output = run_command("git status --porcelain")
	if not success:
		print("❌ Failed to check git status")
		return False

	if output.strip():
		print("❌ Git working directory is not clean. Please commit or stash changes first.")
		print("Uncommitted changes:")
		print(output)
		return False

	return True


def check_git_branch() -> bool:
	"""Check if we're on the master branch."""
	success, branch = run_command("git branch --show-current")
	if not success:
		print("❌ Failed to get current branch")
		return False

	if branch != "master":
		print(f"❌ Not on master branch (currently on: {branch})")
		print("❌ Release must be run from the master branch only!")
		return False

	return True


def get_latest_tag() -> str:
	"""Get the latest git tag."""
	success, tag = run_command("git describe --tags --abbrev=0")
	if not success:
		print("❌ No existing tags found")
		return "None"

	return tag


def check_github_cli() -> bool:
	"""Check if GitHub CLI is available and authenticated."""
	# Check if gh is installed
	success, _ = run_command("gh --version")
	if not success:
		print("❌ GitHub CLI (gh) is not installed")
		print("Please install it from: https://cli.github.com/")
		return False

	# Check if authenticated
	success, _ = run_command("gh auth status")
	if not success:
		print("❌ GitHub CLI is not authenticated")
		print("Please run: gh auth login")
		return False

	print("✅ GitHub CLI is available and authenticated")
	return True


def create_release_branch(version: str) -> bool:
	"""Create a release branch for the version."""
	branch_name = f"release-v{version}"
	success, _ = run_command(f"git checkout -b {branch_name}")
	if not success:
		print(f"❌ Failed to create release branch {branch_name}")
		return False

	print(f"✅ Created release branch {branch_name}")
	return True


def push_and_create_pr(version: str, release_notes: str) -> bool:
	"""Push the release branch and create a GitHub PR."""
	branch_name = f"release-v{version}"

	# Push the branch
	success, _ = run_command(f"git push -u origin {branch_name}")
	if not success:
		print(f"❌ Failed to push branch {branch_name}")
		return False

	print(f"✅ Pushed branch {branch_name}")

	# Create PR using GitHub CLI
	pr_title = f"Release v{version}"
	pr_body = f"Release version {version}"
	if release_notes:
		pr_body += f"\n\n## Release Notes\n{release_notes}"

	pr_body += "\n\nThis PR was automatically created by the release script."

	cmd = ["gh", "pr", "create", "--title", pr_title, "--body", pr_body, "--base", "master", "--head", branch_name]
	success, output = run_command(cmd)
	if not success:
		print(f"❌ Failed to create PR: {output}")
		return False

	print(f"✅ Created PR: {pr_title}")
	print(f"PR URL: {output}")
	return True


def main():
	"""Main release script."""
	print("🚀 SingleStore AI Release Script")
	print("=" * 40)

	# Check if we're in the right directory
	if not Path("src/singlestore_ai/version.py").exists():
		print("❌ Please run this script from the project root directory")
		sys.exit(1)

	# Check git status
	if not check_git_status():
		sys.exit(1)

	# Check git branch
	if not check_git_branch():
		sys.exit(1)

	# Check GitHub CLI
	if not check_github_cli():
		sys.exit(1)

	# Get current version
	current_version = get_current_version()
	print(f"📦 Current version: {current_version}")

	# Get latest tag
	latest_tag = get_latest_tag()
	if latest_tag.startswith("v"):
		latest_tag = latest_tag[1:]  # Remove 'v' prefix
	print(f"🏷️  Latest tag: {latest_tag}")

	# Parse current version
	major, minor, patch = parse_version(current_version)

	# Show version update options
	print("\n📈 Version update options:")
	print(f"1. Patch release: {format_version(major, minor, patch + 1)} (bug fixes)")
	print(f"2. Minor release: {format_version(major, minor + 1, 0)} (new features, backward compatible)")
	print(f"3. Major release: {format_version(major + 1, 0, 0)} (breaking changes)")
	print("4. Custom version")
	print("5. Cancel")

	# Get user choice
	while True:
		choice = input("\nSelect release type (1-5): ").strip()

		if choice == "1":
			new_version = format_version(major, minor, patch + 1)
			release_type = "patch"
			break
		elif choice == "2":
			new_version = format_version(major, minor + 1, 0)
			release_type = "minor"
			break
		elif choice == "3":
			new_version = format_version(major + 1, 0, 0)
			release_type = "major"
			break
		elif choice == "4":
			custom_version = input("Enter custom version (x.y.z): ").strip()
			try:
				parse_version(custom_version)  # Validate format
				new_version = custom_version
				release_type = "custom"
				break
			except Exception:
				print("❌ Invalid version format. Please use x.y.z format.")
				continue
		elif choice == "5":
			print("❌ Release cancelled")
			sys.exit(0)
		else:
			print("❌ Invalid choice. Please select 1-5.")

	# Confirm the release
	print("\n🎯 Release Summary:")
	print(f"   Current version: {current_version}")
	print(f"   New version: {new_version}")
	print(f"   Release type: {release_type}")

	confirm = input("\nProceed with release? (y/N): ").lower()
	if confirm != "y":
		print("❌ Release cancelled")
		sys.exit(0)

	# Get release notes
	print("\n📝 Release notes (optional, press Enter to skip):")
	release_notes = input("Enter release notes: ").strip()

	# Create release branch
	if not create_release_branch(new_version):
		sys.exit(1)

	# Update version file
	update_version_file(new_version)

	# Commit the version change
	success, _ = run_command("git add src/singlestore_ai/version.py")
	if not success:
		print("❌ Failed to stage version.py")
		sys.exit(1)

	commit_message = f"Release v{new_version}"
	if release_notes:
		commit_message += f"\n\n{release_notes}"

	success, _ = run_command(f'git commit -m "{commit_message}"')
	if not success:
		print("❌ Failed to commit version change")
		sys.exit(1)

	print("✅ Committed version change")

	# Push branch and create PR
	if push_and_create_pr(new_version, release_notes):
		print(f"\n🎉 Successfully created release PR for v{new_version}!")
		print("\n📋 Next steps:")
		print("1. Review the PR and get it approved")
		print("2. Merge the PR to master")
		print("3. The CI will automatically create the git tag on merge")
	else:
		print("\n❌ Failed to create PR. You may want to clean up the release branch:")
		print("   git checkout master")
		print(f"   git branch -D release-v{new_version}")
		sys.exit(1)


if __name__ == "__main__":
	main()
