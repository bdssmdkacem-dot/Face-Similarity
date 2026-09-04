#!/usr/bin/env bash
set -euo pipefail

# Generate the standard Flutter Android host for this source-only repository.
# Run from the repository root with a Flutter SDK >= 3.29.
if [ ! -d android ]; then
  flutter create --platforms=android --org com.shabah .
fi

# ONNX Runtime must be retained by R8/ProGuard.
mkdir -p android/app
cat > android/app/proguard-rules.pro <<'EOF'
-keep class ai.onnxruntime.** { *; }
EOF

# Supabase and the model download require network access on Android.
manifest="android/app/src/main/AndroidManifest.xml"
if ! grep -q 'android.permission.INTERNET' "$manifest"; then
  sed -i '/<manifest /a\    <uses-permission android:name="android.permission.INTERNET" />' "$manifest"
fi

echo "Android platform ready."
echo "Next: flutter pub get && flutter analyze && flutter build apk --debug"
