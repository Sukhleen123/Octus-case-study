#!/usr/bin/env bash
# Render docs/architecture.md Mermaid diagram to PNG.
# Requires @mermaid-js/mermaid-cli (mmdc).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT="$REPO_ROOT/docs/architecture.md"
OUTPUT="$REPO_ROOT/docs/architecture.png"

if command -v mmdc &> /dev/null; then
    echo "Rendering architecture diagram to $OUTPUT..."
    mmdc -i "$INPUT" -o "$OUTPUT" --theme default
    echo "Done: $OUTPUT"
else
    echo "mmdc (Mermaid CLI) not found."
    echo ""
    echo "To install:"
    echo "  npm install -g @mermaid-js/mermaid-cli"
    echo ""
    echo "To render manually, paste the Mermaid block from docs/architecture.md into:"
    echo "  https://mermaid.live"
    echo ""
    echo "Or use the VS Code Mermaid extension to preview in-editor."
fi
