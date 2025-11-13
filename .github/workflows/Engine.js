name: Add SupraNova Engine

on:
  workflow_dispatch:  # allows manual run from Actions tab
  schedule:
    - cron: "0 12 * * *"  # optional: runs daily at 12:00 UTC

jobs:
  add-engine:
    runs-on: ubuntu-latest

    permissions:
      contents: write  # required to push commits

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Create engines directory if missing
        run: mkdir -p engines

      - name: Download SupraNova engine binary
        run: |
          wget -O engines/Supranova-ubuntu-latest \
            https://github.com/suprateem-ux/SupraNova-Chess-engine-/releases/download/v1.0.12/Supranova-ubuntu-latest
          chmod +x engines/Supranova-ubuntu-latest

      - name: Commit and push changes
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "41898282+github-actions[bot]@users.noreply.github.com"
          if git diff --quiet && git diff --staged --quiet; then
            echo "No changes to commit"
          else
            git add engines/Supranova-ubuntu-latest
            git commit -m "Update SupraNova engine binary"
            git push
          fi
