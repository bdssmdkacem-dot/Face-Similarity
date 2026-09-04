#!/usr/bin/env bash
set -euo pipefail

# Generate the standard Flutter Android host for this source-only repository.
# Run from the repository root with a Flutter SDK >= 3.29.
flutter create --platforms=android .

# flutter_onnxruntime requires these classes to be retained by R8/ProGuard.
mkdir -p android/app
cat > android/app/proguard-rules.pro <<'EOF'
-keep class ai.onnxruntime.** { *; }
EOF

echo "Android platform generated and ONNX Runtime keep rules installed."
echo "Next: flutter pub get && flutter analyze && flutter build apk --debug"
