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
    return CelebrityMatch(
      celebrityId: map['celebrity_id'] as String,
      name: map['name'] as String,
      similarity: (map['similarity'] as num).toDouble(),
      imageUrl: map['image_url'] as String?,
    );
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
    if (embedding.length != 512) {
      throw ArgumentError.value(
        embedding.length,
        'embedding',
        'Expected exactly 512 dimensions.',
      );
    }

    final result = await _client.rpc(
      'match_celebrity_faces',
      params: {
        'query_embedding': embedding,
        'match_count': limit,
      },
    );

    return (result as List)
        .map((row) => CelebrityMatch.fromMap(
              Map<String, dynamic>.from(row as Map),
            ))
        .toList(growable: false);
  }
}
