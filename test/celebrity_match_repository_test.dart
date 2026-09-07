import 'package:flutter_test/flutter_test.dart';

import 'package:face_similarity/data/celebrity_match_repository.dart';

void main() {
  group('CelebrityMatch.fromMap', () {
    test('parses the canonical RPC contract', () {
      final match = CelebrityMatch.fromMap({
        'celebrity_id': '00000000-0000-0000-0000-000000000001',
        'name': 'Example Celebrity',
        'similarity': 0.8734,
        'image_url': 'https://example.com/portrait.jpg',
      });

      expect(match.name, 'Example Celebrity');
      expect(match.similarity, closeTo(0.8734, 0.00001));
      expect(match.imageUrl, 'https://example.com/portrait.jpg');
    });

    test('accepts the legacy celebrity_name field', () {
      final match = CelebrityMatch.fromMap({
        'celebrity_id': '00000000-0000-0000-0000-000000000002',
        'celebrity_name': 'Legacy Celebrity',
        'similarity': 1.2,
      });

      expect(match.name, 'Legacy Celebrity');
      expect(match.similarity, 1.0);
      expect(match.imageUrl, isNull);
    });

    test('rejects malformed results', () {
      expect(
        () => CelebrityMatch.fromMap({'similarity': 0.5}),
        throwsA(isA<FormatException>()),
      );
      expect(
        () => CelebrityMatch.fromMap({
          'celebrity_id': 'id',
          'name': 'Celebrity',
          'similarity': 'bad',
        }),
        throwsA(isA<FormatException>()),
      );
    });
  });
}
