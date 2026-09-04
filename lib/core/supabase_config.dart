import 'package:flutter/foundation.dart';

class SupabaseConfig {
  const SupabaseConfig._();

  static const url = String.fromEnvironment(
    'SUPABASE_URL',
    defaultValue: 'https://giezarwikczqdwiqvctz.supabase.co',
  );

  // This is the Supabase publishable client key. It is safe to ship in the
  // Flutter app; never put the Supabase secret/service-role key here.
  static const publishableKey = String.fromEnvironment(
    'SUPABASE_PUBLISHABLE_KEY',
    defaultValue: 'sb_publishable_4qRT4GNFdbarqrDdB9tQ9w_wo5vox2i',
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
