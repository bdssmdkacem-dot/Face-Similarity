import 'package:flutter/foundation.dart';

class SupabaseConfig {
  const SupabaseConfig._();

  static const url = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://giezarwikczqdwiqvctz.supabase.co',
  );

  static const publishableKey = String.fromEnvironment(
    'SUPABASE_PUBLISHABLE_KEY',
    defaultValue: '',
  );

  static bool get isConfigured => publishableKey.isNotEmpty;

  static void validate() {
    if (!isConfigured) {
      debugPrint(
        'Supabase publishable key is not configured. '
        'Run with --dart-define=SUPABASE_PUBLISHABLE_KEY=...',
      );
    }
  }
}
