import 'dart:async';

import 'package:supabase_flutter/supabase_flutter.dart';

class CelebrityMatch {
  const CelebrityMatch({
    required this.celebrityId,
    required this.name,
    required this.similarity,
    this.imageUrl,
  });

  final String celebrityId;
  final String name;
  final double similarity;
  final String? imageUrl;

  factory CelebrityMatch.fromMap(Map<String, dynamic> map) {
    final id = map['celebrity_id']?.toString();
    final name = (map['name'] ?? map['celebrity_name'])?.toString();
    final rawSimilarity = map['similarity'];

    if (id == null || id.isEmpty) {
      throw const FormatException('Missing celebrity_id in match result.');
    }
    if (name == null || name.isEmpty) {
      throw const FormatException('Missing celebrity name in match result.');
    }
    if (rawSimilarity is! num || !rawSimilarity.isFinite) {
      throw const FormatException('Invalid similarity in match result.');
    }

    return CelebrityMatch(
      celebrityId: id,
      name: name,
      similarity: rawSimilarity.toDouble().clamp(0.0, 1.0),
      imageUrl: _optionalString(map['image_url']),
    );
  }

  static String? _optionalString(Object? value) {
    final text = value?.toString().trim();
    return text == null || text.isEmpty ? null : text;
  }
}

class CelebrityMatchRepository {
  CelebrityMatchRepository({SupabaseClient? client})
    : _client = client ?? Supabase.instance.client;

  final SupabaseClient _client;

  Future<List<CelebrityMatch>> matchEmbedding(
    List<double> embedding, {
    int limit = 5,
  }) async {
    _validateEmbedding(embedding);
    final matchCount = limit.clamp(1, 20);

    final result = await _client
        .rpc(
          'match_celebrity_faces',
          params: {
            'query_embedding': embedding,
            'match_count': matchCount,
          },
        )
        .timeout(
          const Duration(seconds: 20),
          onTimeout:
              () => throw TimeoutException('البحث عن النتائج تجاوز 20 ثانية.'),
        );

    if (result is! List) {
      throw const FormatException('صيغة نتائج البحث غير صالحة.');
    }

    return result
        .map(
          (row) => CelebrityMatch.fromMap(
            Map<String, dynamic>.from(row as Map),
          ),
        )
        .toList(growable: false);
  }

  void _validateEmbedding(List<double> embedding) {
    if (embedding.length != 512) {
      throw ArgumentError.value(
        embedding.length,
        'embedding',
        'Expected exactly 512 dimensions.',
      );
    }

    for (var i = 0; i < embedding.length; i++) {
      if (!embedding[i].isFinite) {
        throw ArgumentError.value(
          embedding[i],
          'embedding[$i]',
          'Embedding values must be finite.',
        );
      }
    }
  }
}
