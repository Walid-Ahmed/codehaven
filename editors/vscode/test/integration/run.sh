#!/bin/sh
# Try the extension inside the VS Code installed on this computer, with a throwaway profile, so your own settings and
# extensions are not used or changed. Usage: test/integration/run.sh [path/to/codehaven]
# A VS Code window opens for a few seconds while the checks run.
set -e
here=$(cd "$(dirname "$0")/../.." && pwd)
# The VS Code program itself, not the code launcher, so the checks' output and exit code come back here.
app="/Applications/Visual Studio Code.app/Contents/MacOS"
code=${VSCODE:-$(ls "$app/Electron" "$app/Code" 2>/dev/null | head -n 1)}
[ -n "$code" ] || code=$(command -v code)
codehaven=${1:-$(command -v codehaven || true)}
[ -n "$codehaven" ] || { echo "Give the path of the codehaven command, such as .venv/bin/codehaven"; exit 1; }
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir -p "$work/project/pkg" "$work/profile/User"
printf 'def apply_discount(p):\n    return p * 0.9\n\n\ndef checkout_total(p):\n    return apply_discount(p)\n' > "$work/project/pkg/pricing.py"
cat > "$work/profile/User/settings.json" <<JSON
{"codehaven.command": "$codehaven", "codehaven.startOllama": false,
 "codehaven.extraArguments": ["--no-remember", "--no-semantic"],
 "security.workspace.trust.enabled": false, "telemetry.telemetryLevel": "off", "update.mode": "none",
 "extensions.autoCheckUpdates": false, "workbench.startupEditor": "none"}
JSON
"$code" --user-data-dir "$work/profile" --extensions-dir "$work/extensions" --disable-extensions --new-window \
  --extensionDevelopmentPath "$here" --extensionTestsPath "$here/test/integration/index.js" "$work/project"
