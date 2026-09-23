#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/.." && pwd)"
data_dir="${1:-$project_root/data/raw/amazon_lmrrc2021}"
source_uri="s3://amazon-last-mile-challenges/almrrc2021/"

if ! command -v aws >/dev/null 2>&1; then
  echo "AWS CLI is required: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html" >&2
  exit 1
fi

mkdir -p "$data_dir"
AWS_EC2_METADATA_DISABLED=true aws s3 sync \
  "$source_uri" \
  "$data_dir/" \
  --no-sign-request \
  --only-show-errors
